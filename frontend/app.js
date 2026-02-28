/**
 * Zuora Help Agent - Frontend Application
 * Simple chat interface connecting to FastAPI backend
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINT = `${API_BASE_URL}/ask`;

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const questionForm = document.getElementById('questionForm');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const sendButtonText = document.getElementById('sendButtonText');
const sendButtonLoader = document.getElementById('sendButtonLoader');
const statusBar = document.getElementById('statusBar');
const statusText = document.getElementById('statusText');

// State
let conversationId = null;
let isLoading = false;

/**
 * Initialize application
 */
function init() {
    // Generate conversation ID
    conversationId = generateConversationId();

    // Event listeners
    questionForm.addEventListener('submit', handleSubmit);

    // Focus input
    questionInput.focus();

    updateStatus('Ready to answer your questions', 'success');
}

/**
 * Generate unique conversation ID
 */
function generateConversationId() {
    return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    e.preventDefault();

    const question = questionInput.value.trim();
    if (!question || isLoading) return;

    // Clear input
    questionInput.value = '';

    // Add user message
    addMessage(question, 'user');

    // Show loading state
    setLoading(true);
    const typingIndicator = showTypingIndicator();

    try {
        // Call API
        const response = await askQuestion(question);

        // Remove typing indicator
        removeTypingIndicator(typingIndicator);

        // Add bot response
        addMessage(response.answer, 'bot', response.sources);

        updateStatus('Question answered', 'success');
    } catch (error) {
        // Remove typing indicator
        removeTypingIndicator(typingIndicator);

        // Show error
        addErrorMessage(error.message);

        updateStatus('Error: ' + error.message, 'error');
    } finally {
        setLoading(false);
        questionInput.focus();
    }
}

/**
 * Call /ask API endpoint
 */
async function askQuestion(question) {
    updateStatus('Thinking...');

    const requestBody = {
        question: question,
        conversation_id: conversationId,
        context: {},
    };

    const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
            errorData.detail || `API error: ${response.status} ${response.statusText}`
        );
    }

    const data = await response.json();
    return data;
}

/**
 * Add message to chat
 */
function addMessage(text, type = 'bot', sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Add text
    const textP = document.createElement('p');
    textP.textContent = text;
    contentDiv.appendChild(textP);

    // Add sources if present
    if (sources && sources.length > 0) {
        const sourcesDiv = createSourcesElement(sources);
        contentDiv.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);

    // Scroll to bottom
    scrollToBottom();
}

/**
 * Create sources element
 */
function createSourcesElement(sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';

    const title = document.createElement('div');
    title.className = 'sources-title';
    title.textContent = '📚 Sources:';
    sourcesDiv.appendChild(title);

    const sourceList = document.createElement('ul');
    sourceList.className = 'source-list';

    sources.forEach((source, index) => {
        const li = document.createElement('li');
        li.className = 'source-item';

        const link = document.createElement('a');
        link.className = 'source-link';
        link.href = source.url || '#';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';

        // Add icon
        const icon = document.createElement('span');
        icon.className = 'source-icon';
        icon.textContent = '📄';
        link.appendChild(icon);

        // Add text
        const text = document.createElement('span');
        text.textContent = source.title || source.source || `Source ${index + 1}`;
        link.appendChild(text);

        // If no URL, prevent default
        if (!source.url) {
            link.onclick = (e) => e.preventDefault();
            link.style.cursor = 'default';
        }

        li.appendChild(link);
        sourceList.appendChild(li);
    });

    sourcesDiv.appendChild(sourceList);
    return sourcesDiv;
}

/**
 * Add error message
 */
function addErrorMessage(errorText) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message error-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textP = document.createElement('p');
    textP.innerHTML = `<strong>Error:</strong> ${escapeHtml(errorText)}`;
    contentDiv.appendChild(textP);

    const helpP = document.createElement('p');
    helpP.textContent = 'Please try again or check if the backend is running.';
    helpP.style.fontSize = '0.9rem';
    helpP.style.marginTop = '0.5rem';
    contentDiv.appendChild(helpP);

    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'typing-indicator';

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        dot.className = 'typing-dot';
        typingDiv.appendChild(dot);
    }

    messageDiv.appendChild(typingDiv);
    chatContainer.appendChild(messageDiv);

    scrollToBottom();

    return messageDiv;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator(indicator) {
    if (indicator && indicator.parentNode) {
        indicator.parentNode.removeChild(indicator);
    }
}

/**
 * Set loading state
 */
function setLoading(loading) {
    isLoading = loading;

    questionInput.disabled = loading;
    sendButton.disabled = loading;

    if (loading) {
        sendButtonText.classList.add('hidden');
        sendButtonLoader.classList.remove('hidden');
    } else {
        sendButtonText.classList.remove('hidden');
        sendButtonLoader.classList.add('hidden');
    }
}

/**
 * Update status bar
 */
function updateStatus(message, type = '') {
    statusText.textContent = message;
    statusBar.className = 'status-bar';

    if (type) {
        statusBar.classList.add(type);
    }

    // Clear success status after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            if (statusBar.classList.contains('success')) {
                updateStatus('Ready');
            }
        }, 3000);
    }
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Check backend health
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Backend health:', data);
            updateStatus(`Connected to backend (${data.version})`, 'success');
            return true;
        }
    } catch (error) {
        console.warn('Backend health check failed:', error);
        updateStatus('⚠️ Backend not reachable. Start the server at ' + API_BASE_URL, 'error');
        return false;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    init();
    checkBackendHealth();
});
