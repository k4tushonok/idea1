"""
Local prompt optimizer based on textual gradients.

Implements beam search with iterative prompt improvement:
1. Find failures on the training set
2. Generate textual gradients (error causes)
3. Apply gradients to create variants + MC paraphrases
4. Select top candidates via pre-screening and full evaluation
"""

from typing import List, Dict, Optional, Set, Tuple
from copy import deepcopy
import time
import random
from data_structures import Example, PromptNode, TextGradient
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor
from diagnostics import (
    is_enabled,
    prompt_id,
    print_population,
    print_timing,
    scores_summary,
    llm_calls,
)
from config import (
    LOCAL_ITERATIONS_PER_GENERATION,
    MIN_IMPROVEMENT,
    LOCAL_BATCH_SIZE,
    PATIENCE,
    SIMILARITY_THRESHOLD,
    LOCAL_PARENTS_PER_ITERATION,
    MINI_BATCH_RATIO,
    PRE_SCREEN_TOP_K,
    MAX_GRADIENT_PAIRS,
    TRAIN_FAILURE_SAMPLE_SIZE,
    MC_SAMPLES_PER_STEP,
    MAX_EXPANSION_FACTOR,
    REJECT_ON_ERRORS,
)


class LocalOptimizer:
    """Local optimizer: beam search + textual gradients.

    Each iteration: find failures → generate feedback →
    edit prompt → select top candidates into beam.
    """

    def __init__(
        self,
        history_manager: HistoryManager,
        scorer: PromptScorer,
        gradient_generator: TextGradientGenerator,
        prompt_editor: PromptEditor,
        llm,
    ):
        self.history = history_manager
        self.scorer = scorer
        self.gradient_gen = gradient_generator
        self.editor = prompt_editor

        # Local optimization statistics
        self.total_iterations = 0
        self.improvements_count = 0
        self.iteration_stats: List[Dict] = []

        # Evaluation cache to avoid re-evaluating the same prompt
        self._train_outcomes_cache: Dict[str, Tuple] = {}
        self._evaluated_prompts: Set[str] = set()
        self.llm = llm

    @staticmethod
    def _normalize_gradient_text(text: str) -> str:
        """Normalize gradient text for deduplication."""
        return " ".join((text or "").lower().split())

    def _select_gradients(
        self, gradients: List[TextGradient], max_pairs: int
    ) -> List[TextGradient]:
        """Select gradients with maximum coverage of unique failures.

        Greedy algorithm: priority score → new failure coverage →
        total failure set size → analysis text length.
        """
        if max_pairs <= 0 or not gradients:
            return []
        if len(gradients) <= max_pairs:
            return gradients

        unique_gradients: List[TextGradient] = []
        seen_feedback = set()
        for gradient in gradients:
            key = self._normalize_gradient_text(gradient.error_analysis)
            if not key or key in seen_feedback:
                continue
            seen_feedback.add(key)
            unique_gradients.append(gradient)

        pool = unique_gradients or gradients
        if len(pool) <= max_pairs:
            return pool

        selected: List[TextGradient] = []
        covered_failures: Set[str] = set()

        while pool and len(selected) < max_pairs:
            best_idx = 0
            best_key = None

            for idx, gradient in enumerate(pool):
                failure_ids = {ex.input_text for ex in gradient.failure_examples}
                new_coverage = len(failure_ids - covered_failures)
                rank_key = (
                    float(gradient.priority),
                    new_coverage,
                    len(failure_ids),
                    len((gradient.error_analysis or "").strip()),
                )
                if best_key is None or rank_key > best_key:
                    best_key = rank_key
                    best_idx = idx

            chosen = pool.pop(best_idx)
            selected.append(chosen)
            covered_failures.update(ex.input_text for ex in chosen.failure_examples)

        return selected

    @staticmethod
    def _example_search_score(ex: Example) -> float:
        """Score an example for failure ranking (lower = harder failure)."""
        if ex.actual_output is None:
            return 0.0
        if ex.is_numeric_qa_task():
            return (
                0.8 * ex.numeric_exact_match_score() + 0.2 * ex.numeric_token_f1_score()
            )
        if ex.metadata and "all_answers" in ex.metadata:
            return 0.8 * ex.qa_exact_match_score() + 0.2 * ex.qa_token_f1_score()
        if ex.metadata and "references" in ex.metadata and "concepts" in ex.metadata:
            return ex.generation_optimization_score()
        return 1.0 if ex.is_success_for_optimization() else 0.0

    def _select_failure_examples(
        self, failures: List[Example], limit: int
    ) -> List[Example]:
        """Select failures for gradient generation: half hardest, rest sampled uniformly."""
        if len(failures) <= limit:
            return failures

        ranked = sorted(failures, key=self._example_search_score)

        hardest_k = max(1, limit // 2)
        selected = ranked[:hardest_k]
        remainder = ranked[hardest_k:]

        slots_left = limit - len(selected)
        if slots_left <= 0 or not remainder:
            return selected[:limit]

        step = len(remainder) / slots_left
        for i in range(slots_left):
            idx = min(int(i * step), len(remainder) - 1)
            selected.append(remainder[idx])

        return selected[:limit]

    def optimize(
        self,
        starting_node: PromptNode,
        train_examples: List[Example],
        validation_examples: List[Example],
    ) -> PromptNode:
        """Run local beam-search optimization.

        Iteratively improves prompts in the beam through:
          Phase 1-2: gradient generation and candidate creation
          Phase 3:   pre-screening on a mini-batch
          Phase 4:   full evaluation
          Phase 5:   beam update

        Returns the best node from the beam.
        """
        print(f"\n{'='*60}")
        print(f"Starting Local Optimization")
        print(f"{'='*60}\n")

        # Add starting node to history if not already there
        if not self.history.get_node(starting_node.id):
            self.history.add_node(starting_node)

        # Evaluate starting node if not yet evaluated
        if not starting_node.is_evaluated:
            print(f"Evaluating starting node...")
            starting_node = self.scorer.evaluate_node(
                starting_node, validation_examples, execute=True, split="validation"
            )
            self.history.update_node(starting_node.id, starting_node)
            print(f"Starting score: {starting_node.metrics.composite_score():.3f}")

        # Initialise beam search
        current_beam: List[PromptNode] = [starting_node]
        best_score = starting_node.selection_score()
        no_improve_iters = 0

        for iteration in range(LOCAL_ITERATIONS_PER_GENERATION):
            iteration_start_time = time.time()
            calls_before = getattr(self.scorer.llm, "total_api_calls", 0)

            self._train_outcomes_cache.clear()
            self.gradient_gen._cache.clear()
            self.editor._cache.clear()

            print(
                f"\n--- Iteration {iteration + 1}/{LOCAL_ITERATIONS_PER_GENERATION} (no_improve={no_improve_iters}/{PATIENCE}) ---"
            )
            if is_enabled():
                print_population(f"beam state iter={iteration + 1}", current_beam)

            # Rotate mini-batch to protect against overfitting
            mini_batch = self._create_mini_batch(validation_examples, seed=iteration)
            if is_enabled():
                print(
                    f"[diag] mini-batch: {len(mini_batch)}/{len(validation_examples)} examples (seed={iteration})"
                )

            # ================================================================
            # PHASE 1+2: GRADIENT GENERATION AND CANDIDATE CREATION
            # For each beam member:
            #   1. Find failures, sample random errors
            #   2. Obtain gradient feedbacks (<START>/<END>)
            #   3. Apply gradient → new prompts (<START>/<END>)
            #   4. Generate MC synonym/paraphrase variants
            #   5. Merge all candidates
            # ================================================================
            all_candidates: List[PromptNode] = []

            for b_idx, parent in enumerate(current_beam, 1):
                print(
                    f"  Beam member {b_idx}/{len(current_beam)} (score: {parent.selection_score():.3f})"
                )

                failure_examples, success_examples, real_rate = (
                    self._get_train_examples_outcomes(
                        parent, train_examples, iteration=iteration
                    )
                )
                if is_enabled():
                    print(
                        f"[diag] beam[{b_idx}] train outcomes: failures={len(failure_examples)} successes={len(success_examples)}"
                    )
                if not failure_examples:
                    print(
                        f"    No failures — skipping gradient generation for this beam member"
                    )
                    continue

                gradients = self.gradient_gen.generate_gradients_batch(
                    parent.prompt_text, failure_examples, success_examples
                )
                print(f"    Generated {len(gradients)} gradients")

                if not gradients:
                    continue

                selected_gradients = self._select_gradients(
                    gradients, MAX_GRADIENT_PAIRS
                )
                if is_enabled():
                    print(
                        f"[diag] beam[{b_idx}] selected_gradients: "
                        f"{len(selected_gradients)}/{len(gradients)}"
                    )

                new_task_sections: List[PromptNode] = []
                for g_idx, gradient in enumerate(selected_gradients, 1):
                    try:
                        variants = self.editor.generate_variants(
                            parent.prompt_text, gradient, parent_node=parent
                        )
                        if is_enabled():
                            print(
                                f"[diag] beam[{b_idx}] gradient {g_idx}: {len(variants)} variants"
                            )
                        new_task_sections.extend(variants)
                    except Exception as e:
                        print(f"    Error generating variants: {e}")
                        continue

                mc_sampled: List[PromptNode] = []
                if MC_SAMPLES_PER_STEP > 0:
                    for variant in new_task_sections:
                        synonyms = self.editor.generate_synonyms(
                            variant.prompt_text,
                            n=MC_SAMPLES_PER_STEP,
                            parent_node=parent,
                        )
                        mc_sampled.extend(synonyms)
                    # Also generate synonyms for the current parent prompt
                    original_synonyms = self.editor.generate_synonyms(
                        parent.prompt_text,
                        n=MC_SAMPLES_PER_STEP,
                        parent_node=parent,
                    )
                    mc_sampled.extend(original_synonyms)

                combined = new_task_sections + mc_sampled
                combined = list(
                    {prompt_id(n.prompt_text): n for n in combined}.values()
                )  # dedup

                # REJECT_ON_ERRORS: discard candidates that do not
                # improve correction of current errors relative to the parent.
                if combined and REJECT_ON_ERRORS and failure_examples:
                    error_exs = random.sample(
                        failure_examples,
                        min(len(failure_examples), 16),
                    )
                    parent_error_score = self._score_on_error_examples(
                        [parent], error_exs
                    )[0]
                    error_scores = self._score_on_error_examples(combined, error_exs)
                    ranked = sorted(
                        zip(combined, error_scores),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    improved = [
                        (c, s) for c, s in ranked if s > parent_error_score + 1e-8
                    ]
                    if improved:
                        combined = [c for c, _ in improved]
                    else:
                        combined = []
                    if is_enabled():
                        print(
                            f"[diag] beam[{b_idx}] reject_on_errors ranking: "
                            f"parent={parent_error_score:.3f} "
                            f"scored={len(ranked)} kept={len(combined)} "
                            f"scores={scores_summary(error_scores)}"
                        )

                if len(combined) > MAX_EXPANSION_FACTOR:
                    combined = combined[:MAX_EXPANSION_FACTOR]
                    if is_enabled():
                        print(
                            f"[diag] beam[{b_idx}] expansion cap: kept {len(combined)} "
                            f"of {MAX_EXPANSION_FACTOR}"
                        )

                all_candidates.extend(combined)
                print(
                    f"    Beam member {b_idx}: {len(combined)} candidates (variants + synonyms)"
                )

            if not all_candidates:
                print("  ✗ No candidates from any beam member")
                no_improve_iters += 1
                self.total_iterations += 1
                self._record_iteration_stats(
                    iteration, iteration_start_time, calls_before
                )
                continue

            # Deduplicate across candidates and against current beam
            unique_candidates = self._deduplicate_candidates(
                all_candidates, current_beam
            )
            print(f"Total unique candidates after dedup: {len(unique_candidates)}")

            if not unique_candidates:
                print("✗ No unique candidates generated")
                no_improve_iters += 1
                self.total_iterations += 1
                self._record_iteration_stats(
                    iteration, iteration_start_time, calls_before
                )
                continue

            # ================================================================
            # PHASE 3: PRE-SCREENING ON MINI-BATCH
            # Evaluate candidates first on a small subset, then
            # fully evaluate only top-K.
            # ================================================================
            if len(unique_candidates) > PRE_SCREEN_TOP_K:
                print(
                    f"Pre-screening {len(unique_candidates)} candidates on mini-batch ({len(mini_batch)} examples)..."
                )
                pre_scores = self._pre_screen_candidates(unique_candidates, mini_batch)

                ranked = sorted(
                    zip(unique_candidates, pre_scores), key=lambda x: x[1], reverse=True
                )
                top_candidates = [c for c, s in ranked[:PRE_SCREEN_TOP_K]]

                if is_enabled():
                    all_pre = [s for _, s in ranked]
                    print(f"[diag] pre-screen scores: {scores_summary(all_pre)}")

                print(f"Pre-screened to top {len(top_candidates)} candidates")
            else:
                top_candidates = unique_candidates

            # ================================================================
            # PHASE 4: FULL EVALUATION
            # ================================================================
            evaluated_candidates = self._evaluate_candidates(
                top_candidates, validation_examples
            )
            print(f"Evaluated {len(evaluated_candidates)} candidates")

            # Quality threshold: average score across beam
            beam_scores = [n.selection_score() for n in current_beam]
            baseline_score = sum(beam_scores) / len(beam_scores) * 0.95
            eligible_candidates = [
                c for c in evaluated_candidates if c.selection_score() >= baseline_score
            ]
            if len(eligible_candidates) != len(evaluated_candidates):
                print(
                    f"Filtered out {len(evaluated_candidates) - len(eligible_candidates)} candidates "
                    f"below baseline composite {baseline_score:.3f}"
                )

            if eligible_candidates:
                # ================================================================
                # PHASE 5: BEAM UPDATE (top-K by score)
                # ================================================================
                all_for_beam = eligible_candidates + current_beam
                # Deduplicate by prompt text, then top-K
                seen_pids = set()
                unique_beam = []
                for n in sorted(
                    all_for_beam, key=lambda x: x.selection_score(), reverse=True
                ):
                    pid = prompt_id(n.prompt_text)
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        unique_beam.append(n)
                current_beam = unique_beam[:LOCAL_PARENTS_PER_ITERATION]

                best_candidate = current_beam[0]
                candidate_score = best_candidate.selection_score()
                improvement = candidate_score - best_score
                print(
                    f"Best candidate score: {candidate_score:.3f} (Δ {improvement:+.3f})"
                )
                if is_enabled():
                    beam_scores = [n.selection_score() for n in current_beam]
                    print(f"[diag] updated beam scores: {scores_summary(beam_scores)}")

                if candidate_score + 1e-8 >= best_score + MIN_IMPROVEMENT:
                    print(f"✓ Improvement found! Updating beam")
                    best_score = candidate_score
                    no_improve_iters = 0
                    self.improvements_count += 1
                else:
                    print(f"Beam updated but no significant improvement over best")
                    no_improve_iters += 1
            else:
                print("✗ No valid candidates generated")
                no_improve_iters += 1

            self._record_iteration_stats(iteration, iteration_start_time, calls_before)
            self.total_iterations += 1

            # Early stopping
            if no_improve_iters >= PATIENCE:
                print(
                    f"\nEarly stopping after {no_improve_iters} iterations without improvement"
                )
                if is_enabled():
                    print_population("beam at early stop", current_beam)
                break

        print(f"\n{'='*60}")
        print(f"Local Optimization Complete")
        print(f"Final score: {best_score:.3f}")
        print(f"Improvements: {self.improvements_count}")
        print(f"{'='*60}\n")

        return max(current_beam, key=lambda n: n.selection_score())

    def _get_train_examples_outcomes(
        self, node: PromptNode, examples: List[Example], iteration: int = 0
    ) -> Tuple[List[Example], List[Example], float]:
        """Execute the prompt on the training set and split into successes and failures.

        Returns (failures, successes, failure_rate).
        """
        sample_size = min(TRAIN_FAILURE_SAMPLE_SIZE, len(examples))
        if sample_size < len(examples):
            sampled = random.sample(examples, sample_size)
            print(f"Random minibatch: {sample_size}/{len(examples)} train examples")
        else:
            sampled = examples

        eval_examples = [
            Example(
                input_text=ex.input_text,
                expected_output=ex.expected_output,
                metadata=dict(ex.metadata),
            )
            for ex in sampled
        ]
        executed_examples = self.scorer.execute_prompt_batch(
            node.prompt_text, eval_examples
        )

        failures: List[Example] = []
        successes: List[Example] = []
        for ex in executed_examples:
            if ex.is_success_for_optimization():
                successes.append(ex)
            else:
                failures.append(ex)

        real_rate = len(failures) / max(len(executed_examples), 1)
        print(
            f"Train failures: {len(failures)}/{len(executed_examples)} ({real_rate:.1%}) [sampled from {len(examples)}]"
        )

        failures_for_gradient = self._select_failure_examples(
            failures, LOCAL_BATCH_SIZE
        )

        if is_enabled():
            print(
                f"[diag] train outcomes after cap: failures={len(failures_for_gradient)} "
                f"successes={len(successes)} failure_rate={real_rate:.3f}"
            )

        return failures_for_gradient, successes, real_rate

    def _create_mini_batch(
        self, validation_examples: List[Example], seed: int = 0
    ) -> List[Example]:
        """Create a rotating mini-batch for pre-screening.
        Different seeds per iteration prevent overfitting to a fixed subset.
        """
        mini_size = max(5, int(len(validation_examples) * MINI_BATCH_RATIO))
        if mini_size >= len(validation_examples):
            return validation_examples
        rng = random.Random(42 + seed)
        indices = sorted(rng.sample(range(len(validation_examples)), mini_size))
        return [validation_examples[i] for i in indices]

    def _pre_screen_candidates(
        self, candidates: List[PromptNode], mini_batch: List[Example]
    ) -> List[float]:
        """Quick evaluation on a mini-batch for pre-screening.
        Returns a list of composite scores, one per candidate.
        """
        _ps_t0 = time.time()
        _ps_calls0 = llm_calls(self.scorer.llm)
        scores = []
        for i, candidate in enumerate(candidates):
            try:
                metrics = self.scorer.evaluate_prompt(
                    candidate.prompt_text,
                    mini_batch,
                    execute=True,
                    sample=False,
                    stage=1,
                )
                score = metrics.composite_score()
                scores.append(score)
                print(f"  Pre-screen {i+1}/{len(candidates)}: {score:.3f} (stage 1)")
            except Exception as e:
                print(f"  Pre-screen error {i+1}: {e}")
                scores.append(0.0)
        if is_enabled():
            _ps_elapsed = time.time() - _ps_t0
            _ps_calls1 = llm_calls(self.scorer.llm)
            print(
                f"[diag] local pre-screen summary: {len(candidates)} candidates, "
                f"scores={scores_summary(scores)} "
                f"llm_calls={_ps_calls1 - _ps_calls0} time={_ps_elapsed:.2f}s"
            )
        return scores

    def _deduplicate_candidates(
        self, candidates: List[PromptNode], beam: List[PromptNode]
    ) -> List[PromptNode]:
        """Deduplicate candidates against each other and against existing beam members."""
        unique: List[PromptNode] = []
        seen = set()
        beam_pids = {prompt_id(n.prompt_text) for n in beam}

        for c in candidates:
            pid = prompt_id(c.prompt_text)
            if pid in seen or pid in beam_pids:
                continue
            is_dup = any(
                1.0 - self.scorer.calculate_edit_distance(c.prompt_text, u.prompt_text)
                > SIMILARITY_THRESHOLD
                for u in unique
            )
            if not is_dup:
                seen.add(pid)
                unique.append(c)
        return unique

    def _record_iteration_stats(
        self, iteration: int, start_time: float, calls_before: int
    ):
        """Record statistics for the current iteration."""
        iteration_time = time.time() - start_time
        calls_after = getattr(self.scorer.llm, "total_api_calls", 0)
        calls_delta = calls_after - calls_before
        self.iteration_stats.append(
            {
                "iteration": iteration + 1,
                "time": iteration_time,
                "llm_calls": calls_delta,
            }
        )
        print(
            f"Iteration time: {iteration_time:.2f}s — LLM calls: {calls_delta} (total: {calls_after})"
        )
        if is_enabled():
            print_timing(f"local iteration {iteration + 1}", iteration_time)

    def _score_on_error_examples(
        self,
        candidates: List[PromptNode],
        error_examples: List[Example],
    ) -> List[float]:
        """Quick scoring of candidates on error examples.

        For each candidate, execute the prompt on error_examples
        and compute the fraction of corrected errors.
        """
        scores = []
        for c in candidates:
            try:
                eval_exs = [
                    Example(
                        input_text=ex.input_text,
                        expected_output=ex.expected_output,
                        metadata=dict(ex.metadata) if ex.metadata else {},
                    )
                    for ex in error_examples
                ]
                executed = self.scorer.execute_prompt_batch(c.prompt_text, eval_exs)
                mean_score = sum(
                    self._example_search_score(ex) for ex in executed
                ) / max(len(executed), 1)
                scores.append(mean_score)
            except Exception:
                scores.append(0.0)
        return scores

    def _evaluate_candidates(
        self, candidates: List[PromptNode], validation_examples: List[Example]
    ) -> List[PromptNode]:
        """Evaluate candidates on the validation set."""
        evaluated = []

        for i, candidate in enumerate(candidates):
            # Skip if already evaluated
            key = candidate.prompt_text
            if key in self._evaluated_prompts:
                print(
                    f"  Candidate {i+1}/{len(candidates)}: Skipped (already evaluated)"
                )
                continue

            existing = self.history.find_node_by_prompt_text(
                candidate.prompt_text, evaluated_only=True
            )
            if existing is not None:
                print(f"  Candidate {i+1}/{len(candidates)}: Reused cached evaluation")
                candidate.metrics = deepcopy(existing.metrics)
                candidate.metadata = deepcopy(existing.metadata)
                candidate.is_evaluated = True
                candidate.evaluation_examples = deepcopy(existing.evaluation_examples)
                candidate.evaluation_examples_by_split = deepcopy(
                    existing.evaluation_examples_by_split
                )
                self.history.add_node(candidate)
                self._evaluated_prompts.add(key)
                evaluated.append(candidate)
                continue

            print(f"  Evaluating candidate {i+1}/{len(candidates)}...", end=" ")
            if is_enabled():
                print(
                    f"\n[diag] candidate details: node_id={candidate.id} "
                    f"prompt_id={prompt_id(candidate.prompt_text)} len={len(candidate.prompt_text)}"
                )

            candidate = self.scorer.evaluate_node(
                candidate, validation_examples, execute=True, split="validation"
            )
            score = candidate.metrics.composite_score()
            print(f"Score: {score:.3f}")

            self.history.add_node(candidate)
            self._evaluated_prompts.add(key)
            evaluated.append(candidate)

        return evaluated

    def get_statistics(self) -> Dict:
        """Local optimization statistics: iterations, improvements, time, LLM calls."""
        return {
            "total_iterations": self.total_iterations,
            "improvements_count": self.improvements_count,
            "improvement_rate": self.improvements_count / max(self.total_iterations, 1),
            "iteration_stats": self.iteration_stats,
            "avg_iteration_time": (
                (
                    sum(s["time"] for s in self.iteration_stats)
                    / len(self.iteration_stats)
                )
                if self.iteration_stats
                else None
            ),
            "total_llm_calls_by_local": sum(
                s["llm_calls"] for s in self.iteration_stats
            ),
        }
