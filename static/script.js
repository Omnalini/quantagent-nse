function analyzeStock() {
    const stock = document.getElementById("stock").value;

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");

    fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ stock: stock })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loading").classList.add("hidden");
        document.getElementById("result").classList.remove("hidden");
    
        const decision = data.final_decision.decision;
    
        const decisionEl = document.getElementById("decision");
        decisionEl.innerText = decision;
    
        decisionEl.className = "decision-box " + decision.toLowerCase();
    
        document.getElementById("reason").innerText =
            data.final_decision.reason.join(", ");
    
        document.getElementById("indicators").innerText =
            JSON.stringify(data.indicator, null, 2);
    
        document.getElementById("trend").innerText =
            JSON.stringify(data.trend, null, 2);
    
        document.getElementById("pattern").innerText =
            JSON.stringify(data.pattern, null, 2);
    
        document.getElementById("risk").innerText =
            data.final_decision.risk
                ? JSON.stringify(data.final_decision.risk, null, 2)
                : "Not applicable (HOLD decision)";
    })
    
    .catch(error => {
        alert("Error fetching data");
        console.error(error);
    });
}
