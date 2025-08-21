// static/js/script.js (Updated for User GUIDs & New UI)

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const chatHistoryList = document.getElementById("chat-history-list");
    const newChatBtn = document.getElementById("new-chat-btn");
    const messageContainer = document.getElementById("message-container");
    const chatForm = document.getElementById("chat-form");
    const promptInput = document.getElementById("prompt-input");
    const fileAttachBtn = document.getElementById("file-attach-btn");
    const urlAttachBtn = document.getElementById("url-attach-btn");
    const fileInput = document.getElementById("file-input");
    const attachmentPreview = document.getElementById("attachment-preview");
    const menuToggleBtn = document.getElementById("menu-toggle-btn");
    const sidebar = document.querySelector(".sidebar");
    const sidebarOverlay = document.getElementById("sidebar-overlay");
    const welcomeScreen = document.getElementById("welcome-screen"); // NEW

    // --- State ---
    let currentChatId = null;
    let attachedFile = null;
    let attachedUrl = null;

    // --- NEW: User GUID Management ---
    const getUserGuid = () => {
        let userGuid = localStorage.getItem('userGuid');
        if (!userGuid) {
            userGuid = self.crypto.randomUUID();
            localStorage.setItem('userGuid', userGuid);
        }
        return userGuid;
    };
    const userGuid = getUserGuid();

    // --- Mobile Sidebar Logic (no changes) ---
    const toggleSidebar = () => {
        sidebar.classList.toggle("show");
        sidebarOverlay.classList.toggle("active");
    };
    const closeSidebar = () => {
        sidebar.classList.remove("show");
        sidebarOverlay.classList.remove("active");
    };

    // --- Auto-resize Textarea (no changes) ---
    promptInput.addEventListener('input', () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = (promptInput.scrollHeight) + 'px';
    });

    // --- UI/UX Helpers ---
    const showWelcomeScreen = (show) => {
        if (welcomeScreen) {
            welcomeScreen.style.display = show ? 'flex' : 'none';
        }
    };

    // --- Add Copy Buttons to Code Blocks (no changes) ---
    const addCopyButtons = (messageEl) => {
        const codeBlocks = messageEl.querySelectorAll('pre');
        codeBlocks.forEach(block => {
            if (block.querySelector('.copy-btn')) return;
            const header = document.createElement('div');
            header.className = 'code-header';
            const button = document.createElement('button');
            button.className = 'copy-btn';
            button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            button.addEventListener('click', () => {
                const code = block.querySelector('code').innerText;
                navigator.clipboard.writeText(code).then(() => {
                    button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    setTimeout(() => {
                        button.innerHTML = '<i class="fas fa-copy"></i> Copy';
                    }, 2000);
                });
            });
            header.appendChild(button);
            block.prepend(header);
        });
    };

    // --- Chat & Message Rendering (no changes) ---
    const renderMessage = (role, content) => {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}-message`;
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        if (role === 'user') {
            contentDiv.textContent = content;
        } else {
            contentDiv.innerHTML = marked.parse(content);
            const cursor = document.createElement("span");
            cursor.className = "typing-cursor";
            contentDiv.appendChild(cursor);
        }
        messageDiv.appendChild(contentDiv);
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        return contentDiv;
    };

    const updateStreamingMessage = (element, newChunk) => {
        const cursor = element.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
        element.dataset.rawContent = (element.dataset.rawContent || "") + newChunk;
        element.innerHTML = marked.parse(element.dataset.rawContent);
        const newCursor = document.createElement("span");
        newCursor.className = "typing-cursor";
        element.appendChild(newCursor);
    };

    const finalizeStreamingMessage = (element) => {
        const cursor = element.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
        hljs.highlightAll();
        addCopyButtons(element);
    };

    // --- Chat History Management (UPDATED with User GUID Header) ---
    const loadChatHistory = async () => {
        try {
            chatHistoryList.innerHTML = '<li>Loading history...</li>'; // Loading state
            const response = await fetch('/api/history', {
                headers: { 'X-User-GUID': userGuid }
            });
            const history = await response.json();
            chatHistoryList.innerHTML = "";
            if (history.length === 0) {
                 chatHistoryList.innerHTML = '<li>No chats yet.</li>';
            } else {
                history.forEach(chat => {
                    const li = document.createElement("li");
                    li.textContent = chat.title;
                    li.dataset.chatId = chat.id;
                    if (chat.id === currentChatId) li.classList.add("active");
                    li.addEventListener("click", () => loadConversation(chat.id));
                    chatHistoryList.appendChild(li);
                });
            }
        } catch (error) {
            console.error("Error loading chat history:", error);
            chatHistoryList.innerHTML = '<li>Error loading history.</li>';
        }
    };

    const loadConversation = async (chatId) => {
        try {
            const response = await fetch(`/api/history/${chatId}`, {
                headers: { 'X-User-GUID': userGuid }
            });
            if (!response.ok) throw new Error("Conversation not found");

            const conversation = await response.json();
            messageContainer.innerHTML = "";
            showWelcomeScreen(false); // Hide welcome screen
            conversation.messages.forEach(msg => {
                const content = msg.parts.join("\n");
                const role = msg.role === 'model' ? 'bot' : 'user';
                const msgEl = renderMessage(role, content);
                if (role === 'bot') {
                    finalizeStreamingMessage(msgEl);
                }
            });
            currentChatId = chatId;
            document.querySelectorAll('#chat-history-list li').forEach(li => {
                li.classList.toggle('active', li.dataset.chatId === chatId);
            });
            closeSidebar();
        } catch (error) {
            console.error("Error loading conversation:", error);
        }
    };

    // --- Event Handlers ---
    menuToggleBtn.addEventListener("click", toggleSidebar);
    sidebarOverlay.addEventListener("click", closeSidebar);

    newChatBtn.addEventListener("click", () => {
        currentChatId = null;
        messageContainer.innerHTML = "";
        showWelcomeScreen(true); // Show welcome screen
        promptInput.value = "";
        document.querySelectorAll('#chat-history-list li').forEach(li => li.classList.remove('active'));
        resetAttachments();
        closeSidebar();
    });

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const promptText = promptInput.value.trim();
        if (!promptText) return;

        if (messageContainer.contains(welcomeScreen)) {
            showWelcomeScreen(false); // Hide welcome screen on first message
        }

        renderMessage("user", promptText);
        promptInput.value = "";
        promptInput.style.height = 'auto';

        const botMessageElement = renderMessage("bot", "");

        const formData = new FormData();
        formData.append("prompt", promptText);
        formData.append("chat_id", currentChatId);
        if (attachedFile) formData.append("file", attachedFile);
        if (attachedUrl) formData.append("website_url", attachedUrl);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'X-User-GUID': userGuid }, // ADDED USER GUID
                body: formData
            });

            const newChatId = response.headers.get('X-Chat-Id');
            const isNewChat = newChatId && newChatId !== currentChatId;
            if (isNewChat) currentChatId = newChatId;

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                updateStreamingMessage(botMessageElement, chunk);
                messageContainer.scrollTop = messageContainer.scrollHeight;
            }

            finalizeStreamingMessage(botMessageElement);

            if (isNewChat) {
                await loadChatHistory(); // Refresh history list
            }

        } catch (error) {
            console.error("Error during chat:", error);
            botMessageElement.innerHTML = "Sorry, something went wrong. Please try again.";
        } finally {
            resetAttachments();
        }
    });

    // Attachment Logic (no changes)
    fileAttachBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            attachedFile = e.target.files[0]; attachedUrl = null;
            showAttachmentPreview(attachedFile.name, 'file');
        }
    });
    urlAttachBtn.addEventListener('click', () => {
        const url = prompt("Enter a YouTube or Website URL:");
        if (url) {
            attachedUrl = url; attachedFile = null;
            showAttachmentPreview(url, 'url');
        }
    });
    const showAttachmentPreview = (name, type) => {
        attachmentPreview.style.display = 'flex';
        attachmentPreview.innerHTML = `<i class="fas fa-${type === 'file' ? 'file-alt' : 'link'}"></i><span>${name}</span><button id="remove-attachment-btn" type="button">&times;</button>`;
        document.getElementById('remove-attachment-btn').addEventListener('click', resetAttachments);
    };
    const resetAttachments = () => {
        attachedFile = null; attachedUrl = null; fileInput.value = "";
        attachmentPreview.style.display = 'none'; attachmentPreview.innerHTML = '';
    };

    // --- Initial Load ---
    loadChatHistory();
});
