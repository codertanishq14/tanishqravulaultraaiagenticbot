// static/js/script.js
// Updated with PDF Export Feature AND FIX for attachment preview

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const chatHistoryList = document.getElementById("chat-history-list");
    const newChatBtn = document.getElementById("new-chat-btn");
    const messageContainer = document.getElementById("message-container");
    const chatForm = document.getElementById("chat-form");
    const promptInput = document.getElementById("prompt-input");
    const fileAttachBtn = document.getElementById("file-attach-btn");
    const urlAttachBtn = document.getElementById("url-attach-btn");
    const screenShareBtn = document.getElementById("screen-share-btn");
    const fileInput = document.getElementById("file-input");
    const attachmentPreview = document.getElementById("attachment-preview");
    const imageGenBtn = document.getElementById("image-gen-btn");
    const videoGenBtn = document.getElementById("video-gen-btn");
    const generationModal = document.getElementById("generation-modal");
    const modalForm = document.getElementById("modal-form");
    const modalCloseBtn = generationModal.querySelector(".modal-close-btn");
    const menuToggleBtn = document.getElementById("menu-toggle-btn");
    const sidebar = document.querySelector(".sidebar");
    const sidebarOverlay = document.getElementById("sidebar-overlay");
    const userIdManagerDiv = document.getElementById("user-id-manager");
    const largeFileAttachBtn = document.getElementById("large-file-attach-btn");
    const largeFileInput = document.getElementById("large-file-input");
    const userIdModal = document.getElementById("user-id-modal");
    const userIdDisplayArea = document.getElementById("user-id-display-area");
    const newUserIdDisplay = document.getElementById("new-user-id-display");
    const copyUserIdBtn = document.getElementById("copy-user-id-btn");
    const confirmUserIdSavedBtn = document.getElementById("confirm-user-id-saved-btn");
    const userIdChoiceArea = document.getElementById("user-id-choice-area");
    const generateUserIdBtn = document.getElementById("generate-user-id-btn");
    const existingUserIdBtn = document.getElementById("existing-user-id-btn");
    const userIdInputArea = document.getElementById("user-id-input-area");
    const existingUserIdInput = document.getElementById("existing-user-id-input");
    const submitExistingUserIdBtn = document.getElementById("submit-existing-user-id-btn");

    // --- State ---
    let currentChatId = null;
    let attachedFile = null;
    let attachedUrl = null;
    let screenStream = null;
    let currentUserId = null;
    let isSubmitting = false;
    let attachedLargeFiles = [];

    // --- A. Initialization Flow ---
    const init = () => {
        currentUserId = localStorage.getItem('chatHistoryUserId');
        if (currentUserId) {
            initializeApp();
        } else {
            showUserIdModal();
        }
    };
    const initializeApp = () => {
        userIdModal.classList.remove('active');
        loadChatHistory();
        renderUserIdManager();
        addCoreEventListeners();
    };
    const showUserIdModal = () => {
        userIdModal.style.display = 'flex';
        setTimeout(() => userIdModal.classList.add('active'), 10);
    };

    // --- B. Core App Logic & Rendering ---
    const autoResizeTextarea = () => {
        promptInput.style.height = 'auto';
        promptInput.style.height = (promptInput.scrollHeight) + 'px';
    };

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
                    setTimeout(() => { button.innerHTML = '<i class="fas fa-copy"></i> Copy'; }, 2000);
                });
            });
            header.appendChild(button);
            block.prepend(header);
        });
    };

    const renderMessage = (role, content) => {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${role}-message`;
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        if (role === 'user') {
            const p = document.createElement('p');
            p.textContent = content;
            contentDiv.appendChild(p);
        } else {
            contentDiv.innerHTML = marked.parse(content);
        }
        messageDiv.appendChild(contentDiv);
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        return contentDiv;
    };

    const createBotMessageElement = () => {
        const messageDiv = document.createElement("div");
        messageDiv.className = "message bot-message";
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        const cursor = document.createElement("span");
        cursor.className = "typing-cursor";
        contentDiv.appendChild(cursor);
        messageDiv.appendChild(contentDiv);
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        return contentDiv;
    };

    const updateBotMessage = (element, newContent, isStatus = false) => {
        if (isStatus) {
            element.innerHTML = `<p>${newContent}</p><span class="typing-cursor"></span>`;
        } else {
            const currentContent = element.dataset.rawContent || "";
            const newRawContent = currentContent + newContent;
            element.dataset.rawContent = newRawContent;
            element.innerHTML = marked.parse(newRawContent) + '<span class="typing-cursor"></span>';
        }
        messageContainer.scrollTop = messageContainer.scrollHeight;
    };

    const finalizeBotMessage = (element) => {
        const cursor = element.querySelector('.typing-cursor');
        if (cursor) cursor.remove();
        const finalContent = element.dataset.rawContent || "";
        element.innerHTML = marked.parse(finalContent);
        hljs.highlightAll();
        addCopyButtons(element);
    };

    const renderUserIdManager = () => {
        if (!currentUserId) {
            userIdManagerDiv.innerHTML = '';
            return;
        }
        const shortId = `${currentUserId.substring(0, 4)}...${currentUserId.substring(currentUserId.length - 4)}`;
        userIdManagerDiv.innerHTML = `
            <p>User ID: <span>${shortId}</span></p>
            <button id="logout-btn">Change ID</button>
        `;
        document.getElementById('logout-btn').addEventListener('click', () => {
            if (confirm("Are you sure? This will clear your current User ID. Make sure you have it saved!")) {
                localStorage.removeItem('chatHistoryUserId');
                window.location.reload();
            }
        });
    };

    // --- C. Data & API Calls (UPDATED with PDF Export Logic) ---
    const loadChatHistory = async () => {
        if (!currentUserId) return;
        try {
            const response = await fetch('/api/history', {
                headers: { 'X-User-ID': currentUserId }
            });
            if (response.status === 401) {
                console.error("Invalid or missing User ID.");
                localStorage.removeItem('chatHistoryUserId');
                window.location.reload();
                return;
            }
            const history = await response.json();
            chatHistoryList.innerHTML = "";
            history.forEach(chat => {
                const li = document.createElement("li");
                li.dataset.chatId = chat.id;

                const titleSpan = document.createElement("span");
                titleSpan.className = "chat-title";
                titleSpan.textContent = chat.title;
                li.appendChild(titleSpan);

                // ### NEW PDF EXPORT BUTTON ###
                const exportBtn = document.createElement("button");
                exportBtn.className = "export-btn";
                exportBtn.title = "Export to PDF";
                exportBtn.innerHTML = '<i class="fas fa-file-pdf"></i>';
                exportBtn.addEventListener("click", (e) => {
                    e.stopPropagation(); 
                    exportChatToPdf(chat.id, exportBtn);
                });
                li.appendChild(exportBtn);

                if (chat.id === currentChatId) li.classList.add("active");
                li.addEventListener("click", () => loadConversation(chat.id));
                chatHistoryList.appendChild(li);
            });
        } catch (error) { console.error("Error loading chat history:", error); }
    };

    const loadConversation = async (chatId) => {
        if (!currentUserId) return;
        try {
            const response = await fetch(`/api/history/${chatId}`, {
                headers: { 'X-User-ID': currentUserId }
            });
            if (!response.ok) throw new Error("Conversation not found");
            const conversation = await response.json();
            messageContainer.innerHTML = "";
            conversation.messages.forEach(msg => {
                const content = msg.parts.join("\n");
                const role = msg.role === 'model' ? 'bot' : 'user';
                const msgEl = renderMessage(role, content);
                if (role === 'bot') {
                    msgEl.dataset.rawContent = content;
                    finalizeBotMessage(msgEl);
                }
            });
            currentChatId = chatId;
            document.querySelectorAll('#chat-history-list li').forEach(li => {
                li.classList.toggle('active', li.dataset.chatId === chatId);
            });
            closeSidebar();
        } catch (error) { console.error("Error loading conversation:", error); }
    };

    // ### NEW PDF EXPORT FUNCTION ###
    const exportChatToPdf = async (chatId, buttonElement) => {
        const originalIcon = buttonElement.innerHTML;
        buttonElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        buttonElement.disabled = true;

        try {
            const response = await fetch(`/api/history/${chatId}/export`, {
                method: 'GET',
                headers: { 'X-User-ID': currentUserId }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => {
                    return { error: "Server returned a non-JSON response. Is the PDF route missing in app.py?" };
                });
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            const disposition = response.headers.get('content-disposition');
            let filename = `chat-history-${chatId}.pdf`;
            if (disposition && disposition.indexOf('attachment') !== -1) {
                const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                const matches = filenameRegex.exec(disposition);
                if (matches != null && matches[1]) { 
                  filename = matches[1].replace(/['"]/g, '');
                }
            }
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

        } catch (error) {
            console.error('Error exporting PDF:', error);
            alert(`Failed to export PDF: ${error.message}`);
        } finally {
            buttonElement.innerHTML = originalIcon;
            buttonElement.disabled = false;
        }
    };

    // --- D. Event Handlers ---
    function addCoreEventListeners() {
        menuToggleBtn.addEventListener("click", () => { sidebar.classList.toggle("show"); sidebarOverlay.classList.toggle("active"); });
        sidebarOverlay.addEventListener("click", closeSidebar);
        newChatBtn.addEventListener("click", () => {
            currentChatId = null;
            messageContainer.innerHTML = '<div class="message bot-message"><div class="message-content"><p>New chat started. How can I assist you?</p></div></div>';
            promptInput.value = "";
            document.querySelectorAll('#chat-history-list li').forEach(li => li.classList.remove('active'));
            resetAttachments(true);
            closeSidebar();
        });
        chatForm.addEventListener("submit", handleChatSubmit);
        promptInput.addEventListener('input', autoResizeTextarea);
        promptInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chatForm.dispatchEvent(new Event('submit')); } });
        fileAttachBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) { resetAttachments(); attachedFile = e.target.files[0]; showAttachmentPreview(attachedFile.name, 'file'); } });
        largeFileAttachBtn.addEventListener('click', () => largeFileInput.click());
        largeFileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) { resetAttachments(); attachedLargeFiles = Array.from(e.target.files); const previewText = attachedLargeFiles.length > 1 ? `${attachedLargeFiles.length} large files` : attachedLargeFiles[0].name; showAttachmentPreview(previewText, 'fa-file-zipper'); } });
        urlAttachBtn.addEventListener('click', () => { const url = prompt("Enter a YouTube or Website URL:"); if (url) { resetAttachments(true); attachedUrl = url; showAttachmentPreview(url, 'url'); } });
        screenShareBtn.addEventListener("click", handleScreenShareClick);
        imageGenBtn.addEventListener("click", () => showModal('image'));
        videoGenBtn.addEventListener("click", () => showModal('video'));
        modalCloseBtn.addEventListener("click", hideModal);
        generationModal.addEventListener("click", (e) => { if (e.target === generationModal) hideModal(); });
        modalForm.addEventListener("submit", handleMediaFormSubmit);
    }
    generateUserIdBtn.addEventListener('click', async () => { try { const response = await fetch('/api/user/new'); const data = await response.json(); if (data.user_id) { newUserIdDisplay.value = data.user_id; userIdChoiceArea.style.display = 'none'; userIdDisplayArea.style.display = 'block'; } } catch (error) { console.error("Failed to generate user ID", error); alert("Could not generate a new User ID."); } });
    copyUserIdBtn.addEventListener('click', () => { newUserIdDisplay.select(); document.execCommand('copy'); copyUserIdBtn.innerHTML = '<i class="fas fa-check"></i>'; setTimeout(() => { copyUserIdBtn.innerHTML = '<i class="fas fa-copy"></i>'; }, 2000); });
    confirmUserIdSavedBtn.addEventListener('click', () => { const newId = newUserIdDisplay.value; if (newId) { localStorage.setItem('chatHistoryUserId', newId); currentUserId = newId; initializeApp(); } });
    existingUserIdBtn.addEventListener('click', () => { userIdChoiceArea.style.display = 'none'; userIdInputArea.style.display = 'block'; existingUserIdInput.focus(); });
    submitExistingUserIdBtn.addEventListener('click', () => { const existingId = existingUserIdInput.value.trim(); if (/^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$/.test(existingId)) { localStorage.setItem('chatHistoryUserId', existingId); currentUserId = existingId; initializeApp(); } else { alert("Invalid User ID format."); } });

    // --- E. Handler Functions ---
    const handleChatSubmit = async (e) => {
        e.preventDefault(); if (isSubmitting) return;
        const promptText = promptInput.value.trim(); const isLargeFileSubmit = attachedLargeFiles.length > 0;
        if (!promptText && !screenStream && !attachedFile && !attachedUrl && !isLargeFileSubmit) return;
        if (!currentUserId) { showUserIdModal(); return; }
        isSubmitting = true;
        let userMessageForDisplay = promptText;
        if (isLargeFileSubmit) { const fileText = attachedLargeFiles.length > 1 ? `${attachedLargeFiles.length} large files` : attachedLargeFiles[0].name; userMessageForDisplay = promptText ? `${promptText} (context: ${fileText})` : `Question about ${fileText}`; }
        else if (attachedFile) { userMessageForDisplay = promptText ? `${promptText} (File: ${attachedFile.name})` : `Question about ${attachedFile.name}`; }
        const finalPrompt = promptText || (screenStream ? "What do you see?" : "Summarize the content.");
        renderMessage("user", userMessageForDisplay); promptInput.value = ""; autoResizeTextarea();
        const botMessageElement = createBotMessageElement();
        const formData = new FormData(); formData.append("prompt", finalPrompt); formData.append("chat_id", currentChatId || 'null');
        if (isLargeFileSubmit) { formData.append('upload_type', 'large'); attachedLargeFiles.forEach(file => { formData.append('large_files[]', file); }); }
        else if (attachedFile) { formData.append('file', attachedFile); }
        else if (screenStream) { try { const blob = await captureScreenAsBlob(); const screenshotFile = new File([blob], "screenshot.png", { type: "image/png" }); formData.append("file", screenshotFile); } catch (error) { console.error("Failed to capture screen:", error); renderMessage('bot', `Error capturing screen: ${error.message}`); isSubmitting = false; return; } }
        else if (attachedUrl) { formData.append("website_url", attachedUrl); }
        try {
            const response = await fetch('/api/chat', { method: 'POST', headers: { 'X-User-ID': currentUserId }, body: formData });
            if (!response.ok || !response.body) throw new Error(`Server error (${response.status})`);
            const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
            while (true) {
                const { value, done } = await reader.read(); if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n'); buffer = parts.pop();
                for (const part of parts) { if (part.startsWith('data:')) { try { const eventData = JSON.parse(part.substring(5)); if (eventData.type === 'status') { updateBotMessage(botMessageElement, eventData.content, true); } else if (eventData.type === 'chunk') { updateBotMessage(botMessageElement, eventData.content, false); } else if (eventData.type === 'error') { throw new Error(eventData.content); } else if (eventData.type === 'new_chat_info') { currentChatId = eventData.chatId; await loadChatHistory(); document.querySelector(`#chat-history-list li[data-chat-id="${currentChatId}"]`).classList.add('active'); } } catch (e) { console.error("Error parsing SSE event:", e, "Data:", part); } } }
            }
            finalizeBotMessage(botMessageElement);
        } catch (error) { console.error("Chat fetch error:", error); const errorMessage = `Sorry, an error occurred. (Error: ${error.message})`; updateBotMessage(botMessageElement, errorMessage, true); finalizeBotMessage(botMessageElement); }
        finally { resetAttachments(true); isSubmitting = false; }
    };

    // ### THIS FUNCTION IS NOW FIXED ###
    const resetAttachments = (fullReset = false) => {
        if (screenStream) {
            stopScreenSharing();
        }
        attachedFile = null;
        attachedUrl = null;
        attachedLargeFiles = [];
        // The next two lines were the cause of the bug and have been removed.
        // We should not clear the file input's value programmatically here.
        if (fullReset) {
            // We only reset the actual input elements on a fullReset, like after a submission.
            fileInput.value = "";
            largeFileInput.value = "";
            attachmentPreview.style.display = 'none';
            attachmentPreview.innerHTML = '';
        }
    };

    const showAttachmentPreview = (name, typeOrIcon) => {
        attachmentPreview.style.display = 'flex';
        const iconClass = { 'desktop': 'fa-desktop', 'url': 'fa-link', 'file': 'fa-file-alt' }[typeOrIcon] || typeOrIcon;
        const closeHandler = () => { if (typeOrIcon === 'desktop') stopScreenSharing(); resetAttachments(true); };
        attachmentPreview.innerHTML = `<i class="fas ${iconClass}"></i><span>${name}</span><button id="remove-attachment-btn" type="button">&times;</button>`;
        document.getElementById('remove-attachment-btn').addEventListener('click', closeHandler);
    };
    const stopScreenSharing = () => { if (screenStream) { screenStream.getTracks().forEach(track => track.stop()); } screenStream = null; screenShareBtn.classList.remove("active"); if (attachmentPreview.innerHTML.includes('fa-desktop')) { resetAttachments(true); } };
    const handleScreenShareClick = () => (screenStream ? stopScreenSharing() : startScreenSharing());
    const captureScreenAsBlob = () => new Promise((resolve, reject) => { if (!screenStream || !screenStream.active) return reject(new Error('Screen sharing not active.')); const track = screenStream.getVideoTracks()[0]; new ImageCapture(track).grabFrame().then(imageBitmap => { const canvas = document.createElement('canvas'); canvas.width = imageBitmap.width; canvas.height = imageBitmap.height; canvas.getContext('2d').drawImage(imageBitmap, 0, 0); canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Canvas to Blob conversion failed.')), 'image/png'); }).catch(reject); });
    const closeSidebar = () => { sidebar.classList.remove("show"); sidebarOverlay.classList.remove("active"); };
    const handleMediaFormSubmit = async (e) => { e.preventDefault(); const type = document.getElementById("modal-type").value, prompt = document.getElementById("modal-prompt").value.trim(); if (!prompt) return; hideModal(); renderMessage('user', `Generate ${type}: "${prompt}"`); const botMessageElement = createBotMessageElement(); updateBotMessage(botMessageElement, `Working on it! Generating your ${type}...`, true); let url, body; if (type === 'image') { url = '/api/generate/image'; const num_images = parseInt(document.getElementById("modal-num-images").value, 10); body = JSON.stringify({ prompt, num_images }); } else { url = '/api/generate/video'; body = JSON.stringify({ prompt }); } try { const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body }); const result = await response.json(); if (!response.ok || !result.success) throw new Error(result.error || "An unknown error occurred."); let mediaHTML; if (type === 'image') { mediaHTML = 'Here are the generated images:<div class="generated-media-grid">'; result.paths.forEach(path => { mediaHTML += `<a href="${path}" target="_blank"><img src="${path}" alt="${prompt}"></a>`; }); mediaHTML += '</div>'; } else { mediaHTML = `Here is your video:<video controls src="${result.path}"></video>`; } botMessageElement.dataset.rawContent = mediaHTML; finalizeBotMessage(botMessageElement); } catch (error) { console.error(`Error generating ${type}:`, error); const errorMessage = `Sorry, I couldn't generate the ${type}. <br><strong>Error:</strong> ${error.message}`; updateBotMessage(botMessageElement, errorMessage, true); finalizeBotMessage(botMessageElement); } };
    const showModal = (type) => { const modalTitle = generationModal.querySelector("#modal-title"), numImagesGroup = generationModal.querySelector("#modal-num-images-group"); modalForm.reset(); generationModal.querySelector("#modal-type").value = type; modalTitle.textContent = type === 'image' ? "Generate Image(s)" : "Generate Video"; numImagesGroup.style.display = type === 'image' ? 'block' : 'none'; generationModal.style.display = 'flex'; setTimeout(() => generationModal.classList.add('active'), 10); };
    const hideModal = () => { generationModal.classList.remove('active'); setTimeout(() => generationModal.style.display = 'none', 300); };

    // --- F. Start the app ---
    init();
});
