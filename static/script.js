/* ==========================================
   STOCK ADVISORY SYSTEM - JAVASCRIPT
   ========================================== */

/**
 * Analyzes the selected stock by sending a request to the backend API
 * and displays the results in the UI.
 */
function analyzeStock() {
    const stock = document.getElementById("stock").value;
    const loadingEl = document.getElementById("loading");
    const resultEl = document.getElementById("result");
    const analyzeBtn = document.getElementById("analyzeBtn");

    // Show loading state
    loadingEl.classList.remove("hidden");
    resultEl.classList.add("hidden");
    analyzeBtn.disabled = true;

    // Make API request to backend
    fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ stock: stock })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    })
    .then(data => {
        // Hide loading, show results
        loadingEl.classList.add("hidden");
        resultEl.classList.remove("hidden");
        analyzeBtn.disabled = false;

        // Extract decision data
        const decision = data.final_decision.decision;

        // Update decision display
        const decisionEl = document.getElementById("decision");
        decisionEl.innerText = decision;
        decisionEl.className = "decision-box " + decision.toLowerCase();

        // Update reason text
        document.getElementById("reason").innerText = 
            data.final_decision.reason.join(" • ");

        // Update indicator data
        document.getElementById("indicators").innerText = 
            formatJSON(data.indicator);

        // Update trend data
        document.getElementById("trend").innerText = 
            formatJSON(data.trend);

        // Update pattern data
        document.getElementById("pattern").innerText = 
            formatJSON(data.pattern);

        // Update risk management data
        document.getElementById("risk").innerText = 
            data.final_decision.risk
                ? formatJSON(data.final_decision.risk)
                : "Not applicable (HOLD decision)";
    })
    .catch(error => {
        // Handle errors
        loadingEl.classList.add("hidden");
        analyzeBtn.disabled = false;
        
        showError("Unable to fetch analysis data. Please ensure the backend server is running.");
        console.error("Error:", error);
    });
}

/**
 * Formats a JSON object for display with proper indentation.
 * @param {Object} obj - The object to format
 * @returns {string} - Formatted JSON string
 */
function formatJSON(obj) {
    return JSON.stringify(obj, null, 2);
}

/**
 * Displays an error message to the user.
 * @param {string} message - The error message to display
 */
function showError(message) {
    const resultEl = document.getElementById("result");
    resultEl.classList.remove("hidden");
    resultEl.innerHTML = `
        <div class="section" style="text-align: center; border-color: #ef4444;">
            <h3 style="color: #ef4444;">⚠️ Error</h3>
            <p style="color: #94a3b8;">${message}</p>
        </div>
    `;
}

// Initialize: Add keyboard shortcut (Enter key triggers analysis)
document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("stock").addEventListener("keypress", function(e) {
        if (e.key === "Enter") {
            analyzeStock();
        }
    });
});
