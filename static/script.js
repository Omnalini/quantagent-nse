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
    const statusEl = document.getElementById("status");

    // Hide status and results
    statusEl.classList.add("hidden");
    resultEl.classList.add("hidden");
    
    // Show loading state
    loadingEl.classList.remove("hidden");
    analyzeBtn.disabled = true;

    // Animate loading steps
    animateLoadingSteps();

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

        // Show success status
        showStatus("success", `Analysis complete for ${stock}`);

        // Extract decision data
        const decision = data.final_decision.decision;

        // Update decision display
        const decisionEl = document.getElementById("decision");
        decisionEl.innerText = decision;
        decisionEl.className = "decision-box " + decision.toLowerCase();

        // Update reason display with styled items
        renderReasons(data.final_decision.reason);

        // Render visualized data
        renderIndicators(data.indicator);
        renderTrend(data.trend);
        renderPattern(data.pattern);
        renderRisk(data.final_decision.risk, decision);
    })
    .catch(error => {
        // Handle errors
        loadingEl.classList.add("hidden");
        analyzeBtn.disabled = false;
        
        showStatus("error", "Unable to connect to the analysis server");
        showError("Please ensure the backend server is running at http://127.0.0.1:5000");
        console.error("Error:", error);
    });
}

/**
 * Shows a status indicator with the specified type and message.
 */
function showStatus(type, message) {
    const statusEl = document.getElementById("status");
    statusEl.className = `status-indicator ${type}`;
    statusEl.querySelector(".status-text").innerText = message;
    statusEl.classList.remove("hidden");
}

/**
 * Animates the loading steps sequentially.
 */
function animateLoadingSteps() {
    const steps = document.querySelectorAll(".loading-steps .step");
    let currentStep = 0;

    steps.forEach(step => {
        step.classList.remove("active", "done");
    });

    const interval = setInterval(() => {
        if (currentStep > 0) {
            steps[currentStep - 1].classList.remove("active");
            steps[currentStep - 1].classList.add("done");
        }
        
        if (currentStep < steps.length) {
            steps[currentStep].classList.add("active");
            currentStep++;
        } else {
            clearInterval(interval);
        }
    }, 600);
}

/**
 * Renders the reason items in a styled format.
 */
function renderReasons(reasons) {
    const reasonEl = document.getElementById("reason");
    if (!reasons || reasons.length === 0) {
        reasonEl.innerHTML = '<div class="no-data">No reasoning available</div>';
        return;
    }

    reasonEl.innerHTML = reasons.map(reason => 
        `<div class="reason-item">${reason}</div>`
    ).join("");
}

/**
 * Renders the indicator data with visual styling.
 */
function renderIndicators(indicators) {
    const container = document.getElementById("indicators");
    
    if (!indicators || Object.keys(indicators).length === 0) {
        container.innerHTML = '<div class="no-data">No indicator data available</div>';
        return;
    }

    const html = Object.entries(indicators).map(([key, value]) => {
        const signal = getSignalClass(value);
        const displayValue = formatValue(value);
        const displayLabel = formatLabel(key);
        
        return `
            <div class="indicator-item">
                <span class="indicator-label">${displayLabel}</span>
                ${typeof value === 'string' && isSignalValue(value) 
                    ? `<span class="signal-badge ${signal}">${displayValue}</span>`
                    : `<span class="indicator-value ${signal}">${displayValue}</span>`
                }
            </div>
        `;
    }).join("");

    container.innerHTML = html;
}

/**
 * Renders the trend analysis with visual bars.
 */
function renderTrend(trend) {
    const container = document.getElementById("trend");
    
    if (!trend || Object.keys(trend).length === 0) {
        container.innerHTML = '<div class="no-data">No trend data available</div>';
        return;
    }

    const html = Object.entries(trend).map(([key, value]) => {
        const direction = getTrendDirection(value);
        const strength = getTrendStrength(value);
        const displayLabel = formatLabel(key);
        
        return `
            <div class="trend-item">
                <div class="trend-header">
                    <span class="trend-label">${displayLabel}</span>
                    <span class="trend-direction ${direction}">
                        ${getTrendIcon(direction)} ${formatValue(value)}
                    </span>
                </div>
                <div class="trend-bar">
                    <div class="trend-bar-fill ${direction}" style="width: ${strength}%"></div>
                </div>
            </div>
        `;
    }).join("");

    container.innerHTML = html;
}

/**
 * Renders the pattern detection data.
 */
function renderPattern(pattern) {
    const container = document.getElementById("pattern");
    
    if (!pattern || Object.keys(pattern).length === 0) {
        container.innerHTML = '<div class="no-data">No patterns detected</div>';
        return;
    }

    // Handle if pattern is a string
    if (typeof pattern === 'string') {
        container.innerHTML = `
            <div class="pattern-item">
                <div class="pattern-name">${pattern}</div>
            </div>
        `;
        return;
    }

    const html = Object.entries(pattern).map(([key, value]) => {
        const displayLabel = formatLabel(key);
        const confidence = typeof value === 'number' ? value : 75;
        
        return `
            <div class="pattern-item">
                <div class="pattern-name">${displayLabel}</div>
                <div class="pattern-details">
                    <span>${formatValue(value)}</span>
                    ${typeof value !== 'string' ? `
                        <span class="pattern-confidence">
                            Confidence: 
                            <span class="confidence-bar">
                                <span class="confidence-fill" style="width: ${confidence}%"></span>
                            </span>
                        </span>
                    ` : ''}
                </div>
            </div>
        `;
    }).join("");

    container.innerHTML = html;
}

/**
 * Renders the risk management visualization.
 */
function renderRisk(risk, decision) {
    const container = document.getElementById("risk");
    
    if (!risk || decision === "HOLD") {
        container.innerHTML = `
            <div class="risk-container">
                <div class="risk-header">
                    <span class="risk-label">Risk Assessment</span>
                    <span class="risk-level medium">Not Applicable</span>
                </div>
                <p style="color: #64748b; font-size: 0.9rem; text-align: center; padding: 20px 0;">
                    Risk parameters are not computed for HOLD decisions
                </p>
            </div>
        `;
        return;
    }

    const riskLevel = calculateRiskLevel(risk);
    const riskPercent = calculateRiskPercent(risk);

    let detailsHtml = '';
    if (typeof risk === 'object') {
        detailsHtml = Object.entries(risk).map(([key, value]) => `
            <div class="risk-detail-item">
                <span class="risk-detail-label">${formatLabel(key)}</span>
                <span class="risk-detail-value">${formatValue(value)}</span>
            </div>
        `).join("");
    }

    container.innerHTML = `
        <div class="risk-container">
            <div class="risk-header">
                <span class="risk-label">Risk Level</span>
                <span class="risk-level ${riskLevel}">${riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)}</span>
            </div>
            <div class="risk-bar-container">
                <div class="risk-marker" style="left: ${riskPercent}%"></div>
            </div>
            <div class="risk-details">
                ${detailsHtml}
            </div>
        </div>
    `;
}

/**
 * Helper function to determine signal class based on value.
 */
function getSignalClass(value) {
    if (typeof value === 'string') {
        const lower = value.toLowerCase();
        if (lower.includes('bullish') || lower.includes('buy') || lower.includes('up') || lower.includes('positive')) {
            return 'bullish';
        }
        if (lower.includes('bearish') || lower.includes('sell') || lower.includes('down') || lower.includes('negative')) {
            return 'bearish';
        }
    }
    return 'neutral';
}

/**
 * Checks if a value is a signal type (bullish/bearish/neutral).
 */
function isSignalValue(value) {
    if (typeof value !== 'string') return false;
    const lower = value.toLowerCase();
    return lower.includes('bullish') || lower.includes('bearish') || 
           lower.includes('buy') || lower.includes('sell') || 
           lower.includes('neutral') || lower.includes('hold');
}

/**
 * Gets the trend direction from a value.
 */
function getTrendDirection(value) {
    if (typeof value === 'string') {
        const lower = value.toLowerCase();
        if (lower.includes('up') || lower.includes('bullish') || lower.includes('positive')) return 'up';
        if (lower.includes('down') || lower.includes('bearish') || lower.includes('negative')) return 'down';
    }
    if (typeof value === 'number') {
        if (value > 0) return 'up';
        if (value < 0) return 'down';
    }
    return 'sideways';
}

/**
 * Gets the trend strength percentage.
 */
function getTrendStrength(value) {
    if (typeof value === 'number') {
        return Math.min(Math.abs(value), 100);
    }
    return 60; // Default strength for string values
}

/**
 * Gets an icon for trend direction.
 */
function getTrendIcon(direction) {
    switch (direction) {
        case 'up': return '&#8593;';
        case 'down': return '&#8595;';
        default: return '&#8596;';
    }
}

/**
 * Calculates risk level from risk data.
 */
function calculateRiskLevel(risk) {
    if (typeof risk === 'object' && risk.risk_level) {
        return risk.risk_level.toLowerCase();
    }
    // Default calculation based on stop loss distance
    if (risk && risk.stop_loss) {
        const stopLossPercent = parseFloat(risk.stop_loss) || 5;
        if (stopLossPercent < 3) return 'low';
        if (stopLossPercent < 7) return 'medium';
        return 'high';
    }
    return 'medium';
}

/**
 * Calculates risk percentage for the marker position.
 */
function calculateRiskPercent(risk) {
    const level = calculateRiskLevel(risk);
    switch (level) {
        case 'low': return 20;
        case 'medium': return 50;
        case 'high': return 80;
        default: return 50;
    }
}

/**
 * Formats a label by converting snake_case to Title Case.
 */
function formatLabel(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Formats a value for display.
 */
function formatValue(value) {
    if (typeof value === 'number') {
        return value.toFixed(2);
    }
    if (typeof value === 'object') {
        return JSON.stringify(value);
    }
    return String(value);
}

/**
 * Displays an error message to the user.
 */
function showError(message) {
    const resultEl = document.getElementById("result");
    resultEl.classList.remove("hidden");
    resultEl.innerHTML = `
        <div class="section" style="text-align: center; border-color: #ef4444;">
            <h3 style="color: #ef4444; justify-content: center;">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="15" y1="9" x2="9" y2="15"></line>
                    <line x1="9" y1="9" x2="15" y2="15"></line>
                </svg>
                Connection Error
            </h3>
            <p style="color: #94a3b8; margin-top: 12px;">${message}</p>
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
