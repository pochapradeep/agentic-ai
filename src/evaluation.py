"""Evaluation metrics and comparison utilities."""
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
import pandas as pd


def comprehensive_evaluation(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation metrics including RAGAS-style metrics.
    
    Args:
        question: The question asked
        answer: The answer generated
        ground_truth: Ground truth answer
        contexts: List of context strings used
        model_name: Name of the model being evaluated
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['answer_length'] = len(answer)
    metrics['word_count'] = len(answer.split())
    metrics['sentence_count'] = len(re.split(r'[.!?]+', answer))
    metrics['avg_words_per_sentence'] = metrics['word_count'] / max(metrics['sentence_count'], 1)
    
    # Key terms coverage (energy sector specific)
    energy_keywords = [
        'green hydrogen', 'hydrogen', 'renewable', 'energy', 'transition',
        'cost', 'benchmark', 'production', 'India', 'policy', 'framework',
        'electrolyzer', 'LCOH', 'levelized', 'incentive', 'target', '2030',
        'renewable energy', 'solar', 'wind', 'infrastructure', 'challenge',
        'opportunity', 'market', 'investment', 'technology', 'efficiency'
    ]
    
    answer_lower = answer.lower()
    found_keywords = [kw for kw in energy_keywords if kw in answer_lower]
    metrics['key_terms_found'] = len(found_keywords)
    metrics['key_terms_coverage'] = len(found_keywords) / len(energy_keywords)
    
    # Technical terms (numbers, percentages, specific values)
    numbers = re.findall(r'\$?\d+[.,]?\d*[%]?', answer)
    metrics['numerical_data_points'] = len(numbers)
    metrics['has_specific_values'] = 1 if len(numbers) > 0 else 0
    
    # Context usage
    if contexts:
        context_text = ' '.join(contexts[:5]).lower()
        context_words = set(context_text.split())
        answer_words = set(answer_lower.split())
        overlap = len(context_words.intersection(answer_words))
        metrics['context_word_overlap'] = overlap
        metrics['context_usage_ratio'] = overlap / max(len(answer_words), 1)
        
        # Check for direct quotes or paraphrases from context
        context_sentences = re.split(r'[.!?]+', context_text)
        answer_sentences = re.split(r'[.!?]+', answer_lower)
        similar_sentences = 0
        for ans_sent in answer_sentences:
            if len(ans_sent.split()) > 5:
                for ctx_sent in context_sentences:
                    if len(ctx_sent.split()) > 5:
                        ans_words = set(ans_sent.split())
                        ctx_words = set(ctx_sent.split())
                        if len(ans_words.intersection(ctx_words)) / max(len(ans_words), 1) > 0.3:
                            similar_sentences += 1
                            break
        metrics['context_based_sentences'] = similar_sentences
        metrics['context_reliance_ratio'] = similar_sentences / max(len(answer_sentences), 1)
    else:
        metrics['context_word_overlap'] = 0
        metrics['context_usage_ratio'] = 0
        metrics['context_based_sentences'] = 0
        metrics['context_reliance_ratio'] = 0
    
    # Ground truth similarity
    gt_words = set(ground_truth.lower().split())
    answer_words = set(answer_lower.split())
    answer_gt_overlap = len(gt_words.intersection(answer_words))
    metrics['ground_truth_word_overlap'] = answer_gt_overlap
    metrics['ground_truth_similarity'] = answer_gt_overlap / max(len(gt_words), 1)
    
    # Answer structure and quality
    metrics['has_introduction'] = 1 if answer_lower.startswith(('based on', 'according to', 'the document', 'the analysis')) else 0
    metrics['has_conclusion'] = 1 if any(word in answer_lower[-100:] for word in ['conclusion', 'summary', 'overall', 'in summary']) else 0
    metrics['has_structure'] = 1 if any(word in answer_lower for word in ['first', 'second', 'third', 'additionally', 'furthermore', 'however']) else 0
    
    # Completeness
    question_lower = question.lower()
    question_keywords = set(question_lower.split())
    question_answer_overlap = len(question_keywords.intersection(answer_words))
    metrics['question_coverage'] = question_answer_overlap / max(len(question_keywords), 1)
    
    # Specificity score
    specific_terms = ['million', 'tonnes', '2030', 'kg', 'percent', '%', 'dollar', '$', 
                     'policy', 'framework', 'mission', 'incentive', 'target']
    general_terms = ['the', 'is', 'are', 'and', 'or', 'but', 'a', 'an', 'in', 'on', 'at', 'to', 'for']
    specific_count = sum(1 for term in specific_terms if term in answer_lower)
    general_count = sum(1 for term in general_terms if term in answer_lower)
    metrics['specificity_ratio'] = specific_count / max(general_count, 1) if general_count > 0 else specific_count
    
    # Readability
    avg_sentence_length = metrics['avg_words_per_sentence']
    metrics['readability_score'] = max(0, min(100, 100 - (avg_sentence_length - 10) * 2))
    
    # RAGAS-STYLE METRICS
    # Context Precision
    if contexts:
        relevant_contexts = 0
        question_gt_keywords = set(question_lower.split()) | set(ground_truth.lower().split())
        
        for ctx in contexts[:10]:
            ctx_lower = ctx.lower()
            ctx_words = set(ctx_lower.split())
            overlap_ratio = len(question_gt_keywords.intersection(ctx_words)) / max(len(question_gt_keywords), 1)
            if overlap_ratio > 0.1:
                relevant_contexts += 1
        
        metrics['context_precision'] = relevant_contexts / max(len(contexts), 1)
    else:
        metrics['context_precision'] = 0.0
    
    # Context Recall
    if contexts:
        all_context_text = ' '.join(contexts).lower()
        context_words = set(all_context_text.split())
        gt_words_set = set(ground_truth.lower().split())
        
        gt_words_in_context = len(gt_words_set.intersection(context_words))
        metrics['context_recall'] = gt_words_in_context / max(len(gt_words_set), 1)
    else:
        metrics['context_recall'] = 0.0
    
    # Faithfulness
    if contexts:
        all_context_text = ' '.join(contexts).lower()
        answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer_lower) if len(s.strip()) > 10]
        context_sentences = [s.strip() for s in re.split(r'[.!?]+', all_context_text) if len(s.strip()) > 10]
        
        faithful_sentences = 0
        for ans_sent in answer_sentences:
            ans_words = set(ans_sent.split())
            for ctx_sent in context_sentences:
                ctx_words = set(ctx_sent.split())
                overlap = len(ans_words.intersection(ctx_words))
                if overlap / max(len(ans_words), 1) > 0.3:
                    faithful_sentences += 1
                    break
        
        metrics['faithfulness'] = faithful_sentences / max(len(answer_sentences), 1) if answer_sentences else 0.0
    else:
        metrics['faithfulness'] = 0.0
    
    # Answer Correctness
    gt_words_set = set(ground_truth.lower().split())
    answer_words_set = set(answer_lower.split())
    
    word_overlap = len(gt_words_set.intersection(answer_words_set))
    word_precision = word_overlap / max(len(answer_words_set), 1)
    word_recall = word_overlap / max(len(gt_words_set), 1)
    
    if word_precision + word_recall > 0:
        metrics['answer_correctness'] = 2 * (word_precision * word_recall) / (word_precision + word_recall)
    else:
        metrics['answer_correctness'] = 0.0
    
    metrics['answer_precision'] = word_precision
    metrics['answer_recall'] = word_recall
    
    # Key facts coverage
    gt_key_facts = [
        'green hydrogen', 'cost', 'benchmark', 'India', 'policy',
        'electrolyzer', 'renewable', '2030', 'million', 'tonnes'
    ]
    facts_in_answer = sum(1 for fact in gt_key_facts if fact in answer_lower)
    metrics['key_facts_coverage'] = facts_in_answer / len(gt_key_facts)
    
    return metrics


def create_comparison_table(
    baseline_metrics: Dict[str, Any],
    advanced_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """Create a comparison DataFrame from metrics."""
    all_metrics = [
        ('answer_length', 'Answer Length (characters)'),
        ('word_count', 'Word Count'),
        ('sentence_count', 'Sentence Count'),
        ('avg_words_per_sentence', 'Average Words per Sentence'),
        ('key_terms_found', 'Key Terms Found'),
        ('key_terms_coverage', 'Key Terms Coverage'),
        ('numerical_data_points', 'Numerical Data Points'),
        ('has_specific_values', 'Has Specific Values (0/1)'),
        ('key_facts_coverage', 'Key Facts Coverage'),
        ('context_precision', 'Context Precision (RAGAS)'),
        ('context_recall', 'Context Recall (RAGAS)'),
        ('faithfulness', 'Faithfulness (RAGAS)'),
        ('answer_correctness', 'Answer Correctness (RAGAS)'),
        ('answer_precision', 'Answer Precision'),
        ('answer_recall', 'Answer Recall'),
        ('context_word_overlap', 'Context Word Overlap'),
        ('context_usage_ratio', 'Context Usage Ratio'),
        ('context_based_sentences', 'Context-Based Sentences'),
        ('context_reliance_ratio', 'Context Reliance Ratio'),
        ('ground_truth_similarity', 'Ground Truth Similarity'),
        ('question_coverage', 'Question Coverage'),
        ('specificity_ratio', 'Specificity Ratio'),
        ('readability_score', 'Readability Score'),
        ('has_introduction', 'Has Introduction (0/1)'),
        ('has_conclusion', 'Has Conclusion (0/1)'),
        ('has_structure', 'Has Structure (0/1)'),
    ]
    
    comparison_data = {
        'Metric': [],
        'Baseline RAG': [],
        'Deep Thinking RAG': [],
        'Improvement': []
    }
    
    for metric_key, metric_name in all_metrics:
        baseline_val = baseline_metrics.get(metric_key, 0)
        advanced_val = advanced_metrics.get(metric_key, 0)
        
        # Calculate improvement
        if isinstance(baseline_val, (int, float)) and baseline_val > 0:
            improvement = ((advanced_val - baseline_val) / baseline_val) * 100
            improvement_str = f"{improvement:+.1f}%"
        elif isinstance(baseline_val, (int, float)) and baseline_val == 0 and advanced_val > 0:
            improvement_str = "∞ (from 0)"
        else:
            improvement_str = "N/A"
        
        # Format values
        if isinstance(baseline_val, float) and 0 < baseline_val < 1:
            baseline_str = f"{baseline_val:.3f}"
            advanced_str = f"{advanced_val:.3f}"
        elif isinstance(baseline_val, float):
            baseline_str = f"{baseline_val:.1f}"
            advanced_str = f"{advanced_val:.1f}"
        else:
            baseline_str = str(baseline_val)
            advanced_str = str(advanced_val)
        
        comparison_data['Metric'].append(metric_name)
        comparison_data['Baseline RAG'].append(baseline_str)
        comparison_data['Deep Thinking RAG'].append(advanced_str)
        comparison_data['Improvement'].append(improvement_str)
    
    return pd.DataFrame(comparison_data)

