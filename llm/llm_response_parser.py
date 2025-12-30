import re
from typing import List, Dict
from collections import defaultdict
from data_structures import Example, TextGradient, PromptNode, EditOperation, OperationType, OptimizationSource
from typing import Optional

MIN_PROMPT_LENGTH = 20
FALLBACK_ANALYSIS_LENGTH = 500
DEFAULT_PRIORITY = 0.5

SECTION_MARKERS = [
    "## ERROR ANALYSIS",
    "## SUGGESTED DIRECTION",
    "## SPECIFIC SUGGESTIONS",
    "## PRIORITY"
]

CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
PROMPT_BLOCK_RE = re.compile(r"PROMPT:\s*```(.*?)```", re.DOTALL | re.IGNORECASE)
PROMPT_FALLBACK_RE = re.compile(r"PROMPT:\s*(.+)", re.DOTALL | re.IGNORECASE)
CHANGES_RE = re.compile(r"CHANGES MADE:\s*(.+?)(?=OPERATION TYPE|PROMPT|$)", re.DOTALL | re.IGNORECASE)
OPERATION_RE = re.compile(r"OPERATION TYPE:\s*(\w+)", re.IGNORECASE)
LIST_ITEM_RE = re.compile(r"^(?:\d+[.)]\s*|[-*•]\s+)(.+)$")
PRIORITY_RE = re.compile(r"(?:priority|score)?\s*:?\s*([01](?:\.\d+)?)", re.IGNORECASE)
CODE_FENCE_STRIP_RE = re.compile(r"^```.*?\n|\n```$", re.DOTALL)
GRADIENT_SPLIT_RE = re.compile(r"(###\s*GRADIENT.*?)(?=###\s*GRADIENT|\Z)", re.DOTALL | re.IGNORECASE)

class MarkdownParser:
    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        return [
            block.strip()
            for block in CODE_BLOCK_RE.findall(text)
            if block and len(block.strip()) >= MIN_PROMPT_LENGTH
        ]

    @staticmethod
    def strip_code_fences(text: str) -> str:
        return CODE_FENCE_STRIP_RE.sub("", text).strip()

class SectionParser:
    @staticmethod
    def split_by_markers(text: str, markers: List[str], case_sensitive: bool = False) -> Dict[str, str]:
        flags = 0 if case_sensitive else re.IGNORECASE
        found = []

        for marker in sorted(markers, key=len, reverse=True):
            for match in re.finditer(re.escape(marker), text, flags):
                found.append((match.start(), match.end(), marker.upper()))

        found.sort(key=lambda x: x[0])

        sections: Dict[str, str] = {}
        for i, (start, end, marker) in enumerate(found):
            if marker in sections:
                continue  # ignore duplicates
            next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
            sections[marker] = text[end:next_start].strip()

        return sections

    @staticmethod
    def extract_numbered_list(text: str) -> List[str]:
        items: List[str] = []
        current: Optional[str] = None

        for line in text.splitlines():
            line = line.rstrip()
            match = LIST_ITEM_RE.match(line.strip())
            if match:
                if current:
                    items.append(current.strip())
                current = match.group(1)
            elif current and line.startswith(" "):
                current += " " + line.strip()

        if current:
            items.append(current.strip())

        return [i for i in items if len(i) > 5]
    
    @staticmethod
    def extract_priority(text: str) -> float:
        for value in PRIORITY_RE.findall(text):
            try:
                score = float(value)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        return DEFAULT_PRIORITY

class VariantParser:
    @staticmethod
    def extract_prompt(block: str) -> Optional[str]:
        match = PROMPT_BLOCK_RE.search(block)
        if match:
            return match.group(1).strip()
        if block and len(block.strip()) >= MIN_PROMPT_LENGTH:
            return block.strip()
        match = PROMPT_FALLBACK_RE.search(block)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_description(block: str) -> str:
        match = CHANGES_RE.search(block)
        return match.group(1).strip() if match else "Edited based on gradient"

    @staticmethod
    def extract_operation_type(block: str) -> OperationType:
        match = OPERATION_RE.search(block)
        return VariantParser.string_to_operation_type(match.group(1)) if match else OperationType.MODIFY_INSTRUCTION

    @staticmethod
    def string_to_operation_type(value: str) -> OperationType:
        normalized = value.lower().replace(" ", "_")
        return OperationType.__members__.get(normalized.upper(), OperationType.MODIFY_INSTRUCTION)

    @staticmethod
    def parse_variants(response_text: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> List[PromptNode]:
        nodes: List[PromptNode] = []

        for block in MarkdownParser.extract_code_blocks(response_text):
            node = VariantParser.parse_single_variant(block, original_prompt, gradient, parent_node)
            if node:
                nodes.append(node)

        return nodes

    @staticmethod
    def parse_single_variant(block: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> Optional[PromptNode]:
        new_prompt = VariantParser.extract_prompt(block)
        if not new_prompt or len(new_prompt) < MIN_PROMPT_LENGTH:
            return None

        operation = EditOperation(
            operation_type=VariantParser.extract_operation_type(block),
            description=VariantParser.extract_description(block),
            gradient_source=gradient,
            before_snippet=original_prompt + "...",
            after_snippet=new_prompt + "...",
        )

        generation = parent_node.generation + 1 if parent_node else 1

        return PromptNode(
            prompt_text=new_prompt,
            parent_id=parent_node.id if parent_node else None,
            generation=generation,
            source=OptimizationSource.LOCAL,
            operations=[operation],
        )

class GradientParser:
    @staticmethod
    def parse_gradient_response(response_text: str, failure_examples: List[Example], success_examples: List[Example], batch_index: int = None, cluster_name: str = None) -> TextGradient:
        sections = SectionParser.split_by_markers(response_text, SECTION_MARKERS)
        error_analysis = (sections.get("## ERROR ANALYSIS", "").strip() or response_text[:FALLBACK_ANALYSIS_LENGTH])
        suggested_direction = (sections.get("## SUGGESTED DIRECTION", "").strip() or "See error analysis for details")

        specific_suggestions = SectionParser.extract_numbered_list(sections.get("## SPECIFIC SUGGESTIONS", ""))

        priority = min(max(SectionParser.extract_priority(sections.get("## PRIORITY", "")), 0.0), 1.0)  

        gradient = TextGradient(
            failure_examples=failure_examples,
            success_examples=success_examples,
            error_analysis=error_analysis,
            suggested_direction=suggested_direction,
            specific_suggestions=specific_suggestions,
            priority=priority,
        )

        gradient.metadata["batch_index"] = batch_index
        gradient.metadata["cluster"] = cluster_name

        return gradient
    
    @staticmethod
    def _detect_cluster_name(block_text: str, cluster_names: List[str]) -> Optional[str]:
        for name in cluster_names:
            if name.lower() in block_text.lower():
                return name
        return None
    
    @staticmethod
    def parse_batch_response(response_text: str, batches: List[List[Example]], cluster_names: List[str], success_examples: List[Example]) -> List[TextGradient]:
        gradients: List[TextGradient] = []
        blocks = GRADIENT_SPLIT_RE.findall(response_text)

        for idx, block_text in enumerate(blocks):
            cluster_name = GradientParser._detect_cluster_name(block_text, cluster_names)
            batch_index = (
                cluster_names.index(cluster_name)
                if cluster_name in cluster_names
                else idx
            )

            if batch_index >= len(batches):
                continue

            grad = GradientParser.parse_gradient_response(
                response_text=block_text,
                failure_examples=batches[batch_index],
                success_examples=success_examples,
                batch_index=batch_index,
                cluster_name=cluster_name,
            )

            gradients.append(grad)

        return gradients
    
class ClusterParser:
    @staticmethod
    def parse_clusters(response_text: str, failure_examples: List[Example]) -> Dict[str, List[Example]]:
        clusters: Dict[str, List[Example]] = defaultdict(list)
        current_category: Optional[str] = None

        for line in response_text.splitlines():
            line = line.strip()
            if line.startswith("CATEGORY:"):
                current_category = line.replace("CATEGORY:", "").strip()
            elif line.startswith("EXAMPLES:") and current_category:
                indices = [
                    int(i) - 1
                    for i in re.findall(r"\d+", line)
                    if i.isdigit()
                ]
                for idx in indices:
                    if 0 <= idx < len(failure_examples):
                        clusters[current_category].append(failure_examples[idx])

        return dict(clusters) if clusters else {"all": failure_examples}
    
class StrategyParser:
    @staticmethod
    def parse_strategies(response_text: str) -> List[Dict]:
        strategies = []
        strategy_blocks = re.split(r'STRATEGY\s+\d+:', response_text)
        
        for block in strategy_blocks[1:]:
            try:
                type_match = re.search(r'TYPE:\s*(\w+)', block)
                desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=RATIONALE:|SPECIFIC_ACTION:|$)', block, re.DOTALL)
                rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=SPECIFIC_ACTION:|$)', block, re.DOTALL)
                action_match = re.search(r'SPECIFIC_ACTION:\s*(.+?)(?=STRATEGY|$)', block, re.DOTALL)
                
                if type_match and desc_match:
                    strategy = {
                        "type": type_match.group(1).strip().upper(),
                        "description": desc_match.group(1).strip(),
                        "rationale": rationale_match.group(1).strip() if rationale_match else "",
                        "action": action_match.group(1).strip() if action_match else ""
                    }
                    strategies.append(strategy)
            except Exception as e:
                print(f"Error parsing strategy block: {e}")
                continue
        
        return strategies