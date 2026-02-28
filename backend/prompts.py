"""
Prompt Templates for RAG Pipeline
Contains system prompts, user prompts, and prompt construction utilities.
"""

from typing import List, Dict, Optional


# System Prompts
SYSTEM_PROMPT = """You are an expert Zuora product assistant. Your role is to help users understand Zuora's products, features, and best practices.

Guidelines:
- Provide accurate, helpful answers based on the provided documentation
- If you don't know something, say so clearly
- Cite sources when possible
- Keep answers concise but comprehensive
- Use technical terms appropriately but explain them when needed
- If the question is ambiguous, ask for clarification

You have access to Zuora product documentation, API references, and best practice guides."""

ZUORA_EXPERT_PROMPT = """You are a senior Zuora platform expert with deep knowledge of:
- Zuora Billing and Revenue
- Zuora CPQ (Configure, Price, Quote)
- Zuora Collect
- Revenue Recognition
- Subscription management
- API integrations
- Zuora data model and architecture

Your responses should be:
1. Technically accurate
2. Based on official Zuora documentation
3. Practical and actionable
4. Clear for both technical and business users"""


# Prompt Templates
def create_rag_prompt(
    question: str, context_docs: List[Dict], conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Create a RAG prompt with question, context, and conversation history.

    Args:
        question: User's question
        context_docs: Retrieved documents from vector store
        conversation_history: Previous conversation turns

    Returns:
        Formatted prompt string
    """
    # Format context from retrieved documents
    context_text = format_context_docs(context_docs)

    # Format conversation history if available
    history_text = format_conversation_history(conversation_history) if conversation_history else ""

    prompt = f"""Based on the following documentation, please answer the user's question.

Documentation Context:
{context_text}

{history_text}

User Question: {question}

Please provide a clear, accurate answer based on the documentation above. If the documentation doesn't contain enough information to answer the question, say so clearly."""

    return prompt


def format_context_docs(docs: List[Dict]) -> str:
    """
    Format retrieved documents into context string.

    Args:
        docs: List of document dictionaries with 'content' and 'metadata'

    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant documentation found."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("content", "")
        source = doc.get("metadata", {}).get("source", "unknown")
        context_parts.append(f"[Document {i} - Source: {source}]\n{content}\n")

    return "\n".join(context_parts)


def format_conversation_history(history: List[Dict]) -> str:
    """
    Format conversation history for context.

    Args:
        history: List of previous conversation turns

    Returns:
        Formatted history string
    """
    if not history:
        return ""

    history_parts = ["Previous Conversation:"]
    for turn in history[-5:]:  # Include last 5 turns
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_parts.append(f"{role.capitalize()}: {content}")

    return "\n".join(history_parts) + "\n"


def create_summarization_prompt(text: str, max_length: int = 200) -> str:
    """
    Create a prompt for summarizing long text.

    Args:
        text: Text to summarize
        max_length: Maximum length of summary

    Returns:
        Summarization prompt
    """
    return f"""Please provide a concise summary of the following text in no more than {max_length} words:

{text}

Summary:"""


def create_clarification_prompt(question: str, ambiguity: str) -> str:
    """
    Create a prompt to request clarification from the user.

    Args:
        question: Original user question
        ambiguity: Description of what's ambiguous

    Returns:
        Clarification request
    """
    return f"""I need some clarification about your question: "{question}"

{ambiguity}

Could you please provide more details?"""


# Specialized Prompts
PRODUCT_COMPARISON_PROMPT = """Compare the following Zuora products based on the documentation:

Products: {products}

Please provide:
1. Key differences
2. Use cases for each
3. Integration considerations
4. Pricing/licensing differences (if mentioned)

Use the provided documentation to ensure accuracy."""


API_INTEGRATION_PROMPT = """Provide guidance for integrating with Zuora APIs based on this question:

Question: {question}

Please include:
1. Relevant API endpoints
2. Authentication requirements
3. Request/response examples
4. Best practices
5. Common pitfalls to avoid

Base your answer on the official API documentation provided in context."""


TROUBLESHOOTING_PROMPT = """Help troubleshoot this Zuora-related issue:

Issue: {issue}

Please provide:
1. Likely causes
2. Step-by-step debugging approach
3. Common solutions
4. When to contact Zuora support
5. Preventive measures

Use the documentation to suggest verified solutions."""


# Prompt Utilities
def get_prompt_for_intent(intent: str) -> str:
    """
    Get specialized prompt based on detected user intent.

    Args:
        intent: Detected intent (comparison, troubleshooting, integration, etc.)

    Returns:
        Appropriate prompt template
    """
    intent_prompts = {
        "comparison": PRODUCT_COMPARISON_PROMPT,
        "integration": API_INTEGRATION_PROMPT,
        "troubleshooting": TROUBLESHOOTING_PROMPT,
        "general": ZUORA_EXPERT_PROMPT,
    }
    return intent_prompts.get(intent, ZUORA_EXPERT_PROMPT)


def validate_prompt_length(prompt: str, max_tokens: int = 8000) -> bool:
    """
    Validate that prompt doesn't exceed token limits.

    Args:
        prompt: Prompt text
        max_tokens: Maximum allowed tokens (rough estimate: 1 token ≈ 4 chars)

    Returns:
        True if within limits
    """
    estimated_tokens = len(prompt) // 4
    return estimated_tokens <= max_tokens
