let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let waitingForApproval = false;
let currentActiveAgent = null;
let terminalInterval = null;

const AGENT_TERMINAL_LOGS = {
  supervisor: [
    "Analyzing traveler input for security guardrails...",
    "Extracting travel constraints: origin, destination, budget, style...",
    "Checking prompt safety compliance (guardrails validation)...",
    "Supervisor routing travel request to selected specialist agents..."
  ],
  flight_agent: [
    "Activating Flight Specialist Agent...",
    "Connecting to AviationStack MCP server database...",
    "Searching flight routes, operators, and connection durations...",
    "Retrieving airfare estimates and peak-season booking tips...",
    "Compiling recommended flight itineraries..."
  ],
  hotel_agent: [
    "Activating Hotel Specialist Agent...",
    "Launching Tavily search MCP for lodging recommendations...",
    "Analyzing neighborhood ratings, transport links, and review scores...",
    "Filtering boutique and high-value options within budget limits...",
    "Formatting hotel suggestions with rate estimates..."
  ],
  weather_agent: [
    "Activating Weather Specialist Agent...",
    "Extracting destination city parameters...",
    "Querying OpenWeather current conditions & seasonal forecasts...",
    "Evaluating climate trends, rainy season markers, and packing needs...",
    "Synthesizing weather advice for traveler comfort..."
  ],
  budget_agent: [
    "Activating Budget Specialist Agent...",
    "Retrieving pricing models for flights, hotels, and local activities...",
    "Constructing cost categories breakdown...",
    "Evaluating plan feasibility against financial caps...",
    "Calculating budget risk areas and money-saving recommendations..."
  ],
  itinerary_agent: [
    "Activating Itinerary Constructor Agent...",
    "Synthesizing specialists outputs (flights, hotels, weather, budget)...",
    "Building chronological day-by-day sightseeing and activities itinerary...",
    "Adding transit instructions, food spots, and resting notes...",
    "Structuring draft travel plan for human verification..."
  ],
  final_agent: [
    "Activating Final Polisher Agent...",
    "Reviewing human-in-the-loop decisions and revision feedback...",
    "Refining day plans, lodging selections, and flight options...",
    "Applying professional formatting to final sections...",
    "Compiling completed travel planner PDF structure..."
  ]
};

function setPrompt(text) {
  document.getElementById("userInput").value = text;
}

function setFeedback(text) {
  document.getElementById("approvalFeedback").value = text;
}

function logToTerminal(text, type = "info") {
  const consoleEl = document.getElementById("terminalConsole");
  if (!consoleEl) return;
  const lineEl = document.createElement("div");
  lineEl.className = `terminal-line ${type}`;
  
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0];
  
  lineEl.innerHTML = `<span class="timestamp">[${timeStr}]</span> <span class="log-text">${text}</span>`;
  consoleEl.appendChild(lineEl);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function startSimulatedTerminalLogs(agent) {
  stopSimulatedTerminalLogs();
  const logs = AGENT_TERMINAL_LOGS[agent] || ["Processing data..."];
  let logIndex = 0;
  
  // Log first line immediately
  logToTerminal(`[${agent.toUpperCase()}] ${logs[logIndex++]}`, "agent");
  
  terminalInterval = setInterval(() => {
    if (logIndex < logs.length) {
      logToTerminal(`[${agent.toUpperCase()}] ${logs[logIndex++]}`, "agent");
    } else {
      // Loop or stop
      clearInterval(terminalInterval);
    }
  }, 1200);
}

function stopSimulatedTerminalLogs() {
  if (terminalInterval) {
    clearInterval(terminalInterval);
    terminalInterval = null;
  }
}

function resetGraphNodes() {
  const nodes = [
    "node-start", "node-supervisor", "node-flight_agent", 
    "node-hotel_agent", "node-weather_agent", "node-budget_agent", 
    "node-itinerary_agent", "node-human_approval", "node-final_agent"
  ];
  nodes.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.className = id === "node-start" ? "graph-node completed" : "graph-node";
      el.style.opacity = "1";
    }
  });
}

function setNodeState(node, state) {
  const el = document.getElementById(`node-${node}`);
  if (!el) return;
  
  if (state === "active") {
    el.classList.remove("completed", "skipped");
    el.classList.add("active");
  } else if (state === "completed") {
    el.classList.remove("active", "skipped");
    el.classList.add("completed");
  } else if (state === "skipped") {
    el.classList.remove("active", "completed");
    el.classList.add("skipped");
  } else {
    el.classList.remove("active", "completed", "skipped");
  }
}

function updateGraphRouting(selectedAgents) {
  const allSpecialists = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent"];
  allSpecialists.forEach(agent => {
    if (selectedAgents.includes(agent)) {
      setNodeState(agent, "idle");
    } else {
      setNodeState(agent, "skipped");
      logToTerminal(`[SUPERVISOR] Routing bypassed ${agent.replace("_", " ")} (no relevant query constraints).`, "system");
    }
  });
}

function setLoading(isLoading) {
  const sendBtn = document.getElementById("sendBtn");
  const btnText = document.getElementById("btnText");
  const btnLoader = document.getElementById("btnLoader");
  const approveBtn = document.getElementById("approveBtn");
  const reviseBtn = document.getElementById("reviseBtn");
  
  if (sendBtn) sendBtn.disabled = isLoading;
  if (approveBtn) approveBtn.disabled = isLoading;
  if (reviseBtn) reviseBtn.disabled = isLoading;
  
  if (btnText && btnLoader) {
    btnText.classList.toggle("hidden", isLoading);
    btnLoader.classList.toggle("hidden", !isLoading);
  }
}

function showError(message) {
  const errorBox = document.getElementById("errorBox");
  const errorMessage = document.getElementById("errorMessage");
  if (errorBox && errorMessage) {
    errorMessage.textContent = message;
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function hideError() {
  const errorBox = document.getElementById("errorBox");
  if (errorBox) errorBox.classList.add("hidden");
}

function extractSection(markdown, sectionTitle, nextTitles) {
  if (!markdown) return "";
  const lowerMd = markdown.toLowerCase();
  
  // Find where the section title matches
  let matchIndex = -1;
  let titleLength = 0;
  
  // Try matching headers (e.g. "## 1. Trip Summary", "## Trip Summary", "Trip Summary:")
  const patterns = [
    `# ${sectionTitle}`,
    `## ${sectionTitle}`,
    `### ${sectionTitle}`,
    `**${sectionTitle}**`,
    `**[${sectionTitle}]**`,
    `**1. ${sectionTitle}**`,
    `**2. ${sectionTitle}**`,
    `**3. ${sectionTitle}**`,
    `**4. ${sectionTitle}**`,
    `**5. ${sectionTitle}**`,
    `**6. ${sectionTitle}**`,
    `**7. ${sectionTitle}**`,
    `1. ${sectionTitle}`,
    `2. ${sectionTitle}`,
    `3. ${sectionTitle}`,
    `4. ${sectionTitle}`,
    `5. ${sectionTitle}`,
    `6. ${sectionTitle}`,
    `7. ${sectionTitle}`,
    sectionTitle
  ];

  for (const pattern of patterns) {
    const idx = lowerMd.indexOf(pattern.toLowerCase());
    if (idx !== -1) {
      matchIndex = idx;
      titleLength = pattern.length;
      break;
    }
  }

  if (matchIndex === -1) return "";

  // The content starts after the section header line
  const contentStart = markdown.indexOf("\n", matchIndex);
  if (contentStart === -1) return markdown.substring(matchIndex + titleLength);

  // Find the start of the next section
  let nextSectionStart = markdown.length;

  for (const nextTitle of nextTitles) {
    const nextPatterns = [
      `# ${nextTitle}`,
      `## ${nextTitle}`,
      `### ${nextTitle}`,
      `**${nextTitle}**`,
      `**[${nextTitle}]**`,
      `**1. ${nextTitle}**`,
      `**2. ${nextTitle}**`,
      `**3. ${nextTitle}**`,
      `**4. ${nextTitle}**`,
      `**5. ${nextTitle}**`,
      `**6. ${nextTitle}**`,
      `**7. ${nextTitle}**`,
      `1. ${nextTitle}`,
      `2. ${nextTitle}`,
      `3. ${nextTitle}`,
      `4. ${nextTitle}`,
      `5. ${nextTitle}`,
      `6. ${nextTitle}`,
      `7. ${nextTitle}`,
      nextTitle
    ];

    for (const nextPattern of nextPatterns) {
      const idx = lowerMd.indexOf(nextPattern.toLowerCase(), contentStart);
      if (idx !== -1 && idx < nextSectionStart) {
        nextSectionStart = idx;
      }
    }
  }

  return markdown.substring(contentStart, nextSectionStart).trim();
}

function populateTabs(markdown, isDraft = false) {
  latestAnswerMarkdown = markdown || "";
  
  // Render complete raw plan
  const rawBox = document.getElementById("resultBox");
  if (rawBox) {
    rawBox.innerHTML = typeof marked !== "undefined" ? marked.parse(markdown || "") : markdown;
  }
  
  // Parse sections
  const summary = extractSection(markdown, "Trip Summary", ["Flight Information", "Hotel Suggestions", "Accommodation", "Weather Information", "Day-by-Day Itinerary", "Estimated Budget"]);
  const itinerary = extractSection(markdown, "Day-by-Day Itinerary", ["Estimated Budget", "Final Recommendations"]);
  const flights = extractSection(markdown, "Flight Information", ["Hotel Suggestions", "Accommodation", "Weather Information", "Day-by-Day Itinerary", "Estimated Budget"]);
  const hotels = extractSection(markdown, "Hotel Suggestions", ["Weather Information", "Day-by-Day Itinerary", "Estimated Budget", "Final Recommendations"]) 
                 || extractSection(markdown, "Accommodation", ["Weather Information", "Day-by-Day Itinerary", "Estimated Budget", "Final Recommendations"]);
  const weather = extractSection(markdown, "Weather Information", ["Day-by-Day Itinerary", "Estimated Budget", "Final Recommendations"])
                  || extractSection(markdown, "Weather & Packing", ["Day-by-Day Itinerary", "Estimated Budget", "Final Recommendations"]);
  const budget = extractSection(markdown, "Estimated Budget", ["Final Recommendations", "Summary"])
                 || extractSection(markdown, "Budget Analysis", ["Final Recommendations", "Summary"]);
  const recommendations = extractSection(markdown, "Final Recommendations", ["Summary"]);

  // Render to respective tab panels
  const renderPanel = (id, content, fallbackMsg) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (content && content.trim().length > 10) {
      el.innerHTML = typeof marked !== "undefined" ? marked.parse(content) : content;
    } else {
      el.innerHTML = `<div class="no-data-msg">${fallbackMsg}</div>`;
    }
  };

  // Itinerary tab gets Summary + Main Itinerary + Recs
  let itineraryHTML = "";
  if (summary) itineraryHTML += `<h3>Trip Summary</h3>${typeof marked !== "undefined" ? marked.parse(summary) : summary}<hr class="section-divider">`;
  if (itinerary) itineraryHTML += `<h3>Day-by-Day Schedule</h3>${typeof marked !== "undefined" ? marked.parse(itinerary) : itinerary}`;
  if (recommendations) itineraryHTML += `<hr class="section-divider"><h3>Expert Recommendations</h3>${typeof marked !== "undefined" ? marked.parse(recommendations) : recommendations}`;
  
  if (itineraryHTML) {
    const el = document.getElementById("tab-itinerary-render");
    if (el) el.innerHTML = itineraryHTML;
  } else {
    // If splitting failed, display everything in Itinerary tab
    const el = document.getElementById("tab-itinerary-render");
    if (el) el.innerHTML = typeof marked !== "undefined" ? marked.parse(markdown) : markdown;
  }

  renderPanel("tab-flights-render", flights, `<i class="fa-solid fa-plane-slash"></i> No flight information parsed or queried.`);
  renderPanel("tab-hotels-render", hotels, `<i class="fa-solid fa-hotel"></i> No lodging suggestions parsed or queried.`);
  renderPanel("tab-budget-render", budget, `<i class="fa-solid fa-wallet"></i> No budget breakdown parsed or queried.`);
  renderPanel("tab-weather-render", weather, `<i class="fa-solid fa-cloud-sun"></i> No weather analysis parsed or queried.`);
}

function switchTab(tabName) {
  // Reset tab buttons
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => btn.classList.remove("active"));
  
  // Find and activate correct button
  const activeBtn = Array.from(tabBtns).find(btn => btn.getAttribute("onclick").includes(tabName));
  if (activeBtn) activeBtn.classList.add("active");
  
  // Hide all tab contents
  const tabContents = document.querySelectorAll(".tab-content");
  tabContents.forEach(content => content.classList.remove("active"));
  
  // Show target content
  const targetContent = document.getElementById(`tab-${tabName}`);
  if (targetContent) targetContent.classList.add("active");
}

function showWorkflow() {
  document.getElementById("heroScreen").classList.add("hidden");
  document.getElementById("workflowSection").classList.remove("hidden");
}

function updateWorkflowUI(data) {
  const reasonEl = document.getElementById("supervisorReasoning");
  if (reasonEl && data.supervisor_reasoning) {
    reasonEl.textContent = data.supervisor_reasoning;
  }
  
  const badge = document.getElementById("guardrailBadge");
  if (badge) {
    const passed = data.guardrail_allowed !== false;
    badge.textContent = passed ? "Guardrails Passed" : "Guardrails Blocked";
    badge.className = passed ? "guardrail-badge passed" : "guardrail-badge blocked";
    badge.innerHTML = passed 
      ? `<i class="fa-solid fa-shield-halved"></i> Guardrails Passed` 
      : `<i class="fa-solid fa-triangle-exclamation"></i> Guardrails Blocked`;
  }
}

function showResult(answer, threadId, isDraft = false, llmCalls = 0) {
  const resultSection = document.getElementById("resultSection");
  const badge = document.getElementById("resultStatusBadge");
  const title = document.getElementById("resultTitle");
  const threadInfo = document.getElementById("threadInfo");
  const llmCallsCount = document.getElementById("llmCallsCount");
  
  if (badge) {
    badge.textContent = isDraft ? "DRAFT PLAN" : "POLISHED PLAN";
    badge.className = isDraft ? "badge-accent draft" : "badge-accent final";
  }
  
  if (title) {
    title.textContent = isDraft ? "Your Draft Itinerary" : "Your Polished Travel Plan";
  }
  
  if (threadInfo) threadInfo.innerHTML = `<i class="fa-solid fa-fingerprint"></i> Trip ID: <strong>${threadId}</strong>`;
  if (llmCallsCount) llmCallsCount.innerHTML = `<i class="fa-solid fa-microchip"></i> LLM Invocations: <strong>${llmCalls}</strong>`;
  
  populateTabs(answer, isDraft);
  switchTab("itinerary");
  
  if (resultSection) {
    resultSection.classList.remove("hidden");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function showApproval(data) {
  waitingForApproval = true;
  const card = document.getElementById("approvalSection");
  const reqText = document.getElementById("approvalRequest");
  if (reqText) reqText.textContent = data.approval_request || "Please review the draft plan and provide approval or refinement details.";
  if (card) {
    card.classList.remove("hidden");
    card.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function hideApproval() {
  waitingForApproval = false;
  const card = document.getElementById("approvalSection");
  if (card) card.classList.add("hidden");
  
  const feedbackInput = document.getElementById("approvalFeedback");
  if (feedbackInput) feedbackInput.value = "";
}

async function sendMessage() {
  hideError();
  if (waitingForApproval) {
    showError("Please approve or revise the current draft itinerary before starting a new planning session.");
    return;
  }
  
  const message = document.getElementById("userInput").value.trim();
  if (!message) {
    showError("Please describe your travel request first.");
    return;
  }
  
  showWorkflow();
  resetGraphNodes();
  
  const consoleEl = document.getElementById("terminalConsole");
  if (consoleEl) consoleEl.innerHTML = "";
  
  logToTerminal("Connecting to TripPilot AI server...", "system");
  logToTerminal(`Initializing new session for message: "${message.substring(0, 50)}..."`, "info");
  
  setLoading(true);
  setNodeState("supervisor", "active");
  startSimulatedTerminalLogs("supervisor");
  
  try {
    const response = await fetch("/api/travel/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: currentThreadId })
    });
    
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Internal Server Error.");
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();
      
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.substring(6).trim();
          if (!jsonStr) continue;
          
          try {
            const event = JSON.parse(jsonStr);
            await processSSEEvent(event);
          } catch (e) {
            console.error("Failed to parse SSE JSON:", jsonStr, e);
          }
        }
      }
    }
  } catch (error) {
    console.error(error);
    logToTerminal(`[CRITICAL] System failure: ${error.message}`, "error");
    showError(error.message);
  } finally {
    setLoading(false);
    stopSimulatedTerminalLogs();
  }
}

async function submitApproval(approved) {
  hideError();
  if (!currentThreadId || !waitingForApproval) {
    showError("There is no active draft itinerary awaiting review.");
    return;
  }
  
  const feedback = document.getElementById("approvalFeedback").value.trim();
  if (!approved && !feedback) {
    showError("Please enter revision feedback explaining what changes you would like to make.");
    const input = document.getElementById("approvalFeedback");
    if (input) input.focus();
    return;
  }
  
  setLoading(true);
  logToTerminal(approved ? "Submitting Traveler Approval..." : "Sending Revision Feedback to agent crew...", "system");
  setNodeState("human_approval", "completed");
  setNodeState("final_agent", "active");
  startSimulatedTerminalLogs("final_agent");
  
  try {
    const response = await fetch("/api/travel/approve/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: currentThreadId, approved, feedback })
    });
    
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Failed to submit approval.");
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();
      
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.substring(6).trim();
          if (!jsonStr) continue;
          
          try {
            const event = JSON.parse(jsonStr);
            await processSSEEvent(event);
          } catch (e) {
            console.error("Failed to parse SSE JSON:", jsonStr, e);
          }
        }
      }
    }
    
    hideApproval();
  } catch (error) {
    console.error(error);
    logToTerminal(`[CRITICAL] Approval error: ${error.message}`, "error");
    showError(error.message);
  } finally {
    setLoading(false);
    stopSimulatedTerminalLogs();
  }
}

async function processSSEEvent(event) {
  switch (event.type) {
    case "start":
      currentThreadId = event.thread_id;
      localStorage.setItem("travel_thread_id", currentThreadId);
      logToTerminal(event.message, "system");
      break;
      
    case "node_complete":
      const node = event.node;
      logToTerminal(event.message, "success");
      setNodeState(node, "completed");
      stopSimulatedTerminalLogs();
      
      // If supervisor finishes, it selects agents to run
      if (node === "supervisor") {
        updateWorkflowUI(event.data);
        if (event.data.selected_agents) {
          updateGraphRouting(event.data.selected_agents);
          logToTerminal(`Supervisor routed tasks: [${event.data.selected_agents.join(", ")}]`, "info");
          // Activate the first agent in the list
          const firstAgent = event.data.selected_agents[0];
          if (firstAgent) {
            setNodeState(firstAgent, "active");
            startSimulatedTerminalLogs(firstAgent);
          }
        }
      } else if (node === "flight_agent") {
        // Prepare next node in list
        const next = getNextAgent("flight_agent");
        if (next) {
          setNodeState(next, "active");
          startSimulatedTerminalLogs(next);
        }
      } else if (node === "hotel_agent") {
        const next = getNextAgent("hotel_agent");
        if (next) {
          setNodeState(next, "active");
          startSimulatedTerminalLogs(next);
        }
      } else if (node === "weather_agent") {
        const next = getNextAgent("weather_agent");
        if (next) {
          setNodeState(next, "active");
          startSimulatedTerminalLogs(next);
        }
      } else if (node === "budget_agent") {
        const next = getNextAgent("budget_agent");
        if (next) {
          setNodeState(next, "active");
          startSimulatedTerminalLogs(next);
        }
      } else if (node === "itinerary_agent") {
        setNodeState("human_approval", "active");
      }
      break;
      
    case "interrupt":
      logToTerminal(event.message, "system");
      setNodeState("itinerary_agent", "completed");
      setNodeState("human_approval", "active");
      showResult(event.data.itinerary, event.data.thread_id, true, 5);
      showApproval(event.data);
      break;
      
    case "complete":
      logToTerminal(event.message, "success");
      setNodeState("final_agent", "completed");
      setNodeState("human_approval", "completed");
      
      // Update result dashboard with final response
      const res = event.data;
      showResult(res.answer || res.final_response, res.thread_id, false, res.llm_calls);
      break;
      
    case "resume":
      logToTerminal(event.message, "system");
      break;
      
    case "error":
      logToTerminal(`[ERROR] ${event.message}`, "error");
      showError(event.message);
      break;
  }
}

function getNextAgent(current) {
  // Find which agents are active in our configuration
  const allNodes = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"];
  const currentIdx = allNodes.indexOf(current);
  
  for (let i = currentIdx + 1; i < allNodes.length; i++) {
    const el = document.getElementById(`node-${allNodes[i]}`);
    // If the node exists and is not marked as skipped, we can run it
    if (el && !el.classList.contains("skipped")) {
      return allNodes[i];
    }
  }
  return null;
}

function copyResult() {
  const text = document.getElementById("resultBox").innerText;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const button = document.querySelector(".copy-btn");
    const original = button.innerHTML;
    button.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
    setTimeout(() => { button.innerHTML = original; }, 1400);
  }).catch(() => showError("Could not copy itinerary."));
}

function downloadPDF() {
  if (!latestAnswerMarkdown || typeof html2pdf === "undefined") {
    return showError("No travel plan available to generate PDF.");
  }
  const button = document.querySelector(".download-btn");
  const original = button.innerHTML;
  button.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Building PDF...`;
  button.disabled = true;
  
  const options = {
    margin: 0.5,
    filename: `trippilot-${currentThreadId || "itinerary"}.pdf`,
    image: { type: "jpeg", quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
    jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
    pagebreak: { mode: ["avoid-all", "css", "legacy"] }
  };
  
  html2pdf()
    .set(options)
    .from(document.getElementById("pdfContent"))
    .save()
    .then(() => {
      button.innerHTML = original;
      button.disabled = false;
    })
    .catch(() => {
      button.innerHTML = original;
      button.disabled = false;
      showError("Could not compile PDF download.");
    });
}

// Add shortcut for launch
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    sendMessage();
  }
});
