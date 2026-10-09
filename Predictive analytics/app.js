// Telecom Predictive & ML Forecasting Engine Frontend Application Logic

let mlData = null;
let chartGrowthInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  fetchPredictionsData();
  setupEventListeners();
});

function initTabs() {
  const navButtons = document.querySelectorAll(".nav-item button");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const headerTitle = document.getElementById("header-title");

  const titlesMap = {
    "tab-overview": "Predictive Intelligence Overview",
    "tab-growth": "Subscriber Growth & Scenario Forecasting",
    "tab-churn": "Customer Churn Risk & Intervention Center",
    "tab-demand": "Network Data Usage Demand Forecaster",
    "tab-nbo": "Expected Recharge & Next-Best-Offer (NBO)"
  };

  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");

      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(pane => pane.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(tabId).classList.add("active");

      if (titlesMap[tabId]) {
        headerTitle.textContent = titlesMap[tabId];
      }
    });
  });
}

async function fetchPredictionsData() {
  try {
    const response = await fetch("predictions_data.json");
    mlData = await response.json();
    console.log("Loaded ML Predictions Data:", mlData);

    renderKPIs();
    renderOverviewCharts();
    renderGrowthForecastChart(0);
    renderChurnTable();
    renderNetworkDemandTab();

    // Default load first customer for NBO tab
    if (mlData.customers && mlData.customers.length > 0) {
      inspectCustomerNBO(mlData.customers[0].customer_id);
    }
  } catch (err) {
    console.error("Failed to load predictions data:", err);
  }
}

function renderKPIs() {
  const summary = mlData.summary;
  document.getElementById("kpi-active-subs").textContent = summary.active_customers.toLocaleString();
  document.getElementById("kpi-churn-rate").textContent = summary.avg_churn_probability + "%";
  document.getElementById("kpi-high-risk-count").textContent = summary.high_risk_churn_count + " High risk churners";
  document.getElementById("kpi-revenue").textContent = "₹" + Math.round(summary.total_revenue).toLocaleString();
  document.getElementById("kpi-data-tb").textContent = summary.total_data_tb.toLocaleString() + " TB";
}

function renderOverviewCharts() {
  // RFM Segments Chart
  const rfmCounts = mlData.summary.rfm_segment_counts;
  const ctxRfm = document.getElementById("chart-rfm").getContext("2d");
  
  new Chart(ctxRfm, {
    type: "doughnut",
    data: {
      labels: Object.keys(rfmCounts),
      datasets: [{
        data: Object.values(rfmCounts),
        backgroundColor: ["#00F2FE", "#4FACFE", "#7F00FF", "#FF4B4B"],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#94A3B8", font: { family: "Inter" } } }
      }
    }
  });

  // Feature Importance Chart
  const featImp = mlData.feature_importances;
  const ctxFeat = document.getElementById("chart-features").getContext("2d");

  new Chart(ctxFeat, {
    type: "bar",
    data: {
      labels: Object.keys(featImp).map(k => k.replace("_", " ")),
      datasets: [{
        label: "Feature Importance Score",
        data: Object.values(featImp),
        backgroundColor: "#00F2FE",
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#94A3B8" }, grid: { display: false } }
      }
    }
  });
}

function renderGrowthForecastChart(boostPct = 0) {
  const forecast = mlData.subscriber_forecast;
  const horizonDays = parseInt(document.getElementById("sim-horizon-slider").value);

  const slicedDates = forecast.dates.slice(0, horizonDays);
  const baseActive = forecast.projected_active.slice(0, horizonDays);

  // Apply marketing boost calculation
  const boostedActive = baseActive.map((val, idx) => Math.round(val * (1 + (boostPct / 100) * (idx / horizonDays))));
  const upper = forecast.upper_bound.slice(0, horizonDays).map((val, idx) => Math.round(val * (1 + (boostPct / 100) * (idx / horizonDays))));
  const lower = forecast.lower_bound.slice(0, horizonDays);

  const ctxGrowth = document.getElementById("chart-growth").getContext("2d");

  if (chartGrowthInstance) {
    chartGrowthInstance.destroy();
  }

  chartGrowthInstance = new Chart(ctxGrowth, {
    type: "line",
    data: {
      labels: slicedDates,
      datasets: [
        {
          label: "Projected Active Subscribers",
          data: boostedActive,
          borderColor: "#00F2FE",
          borderWidth: 3,
          tension: 0.3,
          fill: false
        },
        {
          label: "Upper 95% Confidence Limit",
          data: upper,
          borderColor: "rgba(79, 172, 254, 0.4)",
          borderWidth: 1,
          borderDash: [5, 5],
          fill: false
        },
        {
          label: "Lower 95% Confidence Limit",
          data: lower,
          borderColor: "rgba(255, 75, 75, 0.4)",
          borderWidth: 1,
          borderDash: [5, 5],
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#94A3B8" } }
      },
      scales: {
        x: { ticks: { color: "#94A3B8", maxTicksLimit: 12 }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });
}

function renderChurnTable() {
  const tbody = document.getElementById("churn-table-body");
  const searchQuery = document.getElementById("search-customer").value.toLowerCase();
  const filterRisk = document.getElementById("filter-risk").value;

  tbody.innerHTML = "";

  const filtered = mlData.customers.filter(c => {
    const matchesSearch = c.name.toLowerCase().includes(searchQuery) ||
                          c.customer_id.toLowerCase().includes(searchQuery) ||
                          c.circle.toLowerCase().includes(searchQuery);
    const matchesRisk = (filterRisk === "All") || (c.risk_level === filterRisk);
    return matchesSearch && matchesRisk;
  }).slice(0, 50); // Show top 50 matches for smooth performance

  filtered.forEach(c => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${c.customer_id}</strong></td>
      <td>${c.name}</td>
      <td>${c.circle}</td>
      <td>${c.recency_days}d ago</td>
      <td>${c.current_plan}</td>
      <td><strong>${c.churn_probability}%</strong></td>
      <td><span class="risk-badge risk-${c.risk_level}">${c.risk_level} (${c.churn_probability}%)</span></td>
      <td><span class="segment-badge">${c.rfm_segment}</span></td>
      <td>
        <button class="action-btn" onclick="openInterventionModal('${c.customer_id}', '${c.name}', ${c.churn_probability})">
          Intervene
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderNetworkDemandTab() {
  const circles = mlData.circle_demand;
  const labels = circles.map(c => c.circle);
  const dataTB = circles.map(c => c.projected_monthly_tb);

  const ctxDemand = document.getElementById("chart-data-circle").getContext("2d");
  new Chart(ctxDemand, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Projected Monthly Throughput (TB)",
        data: dataTB,
        backgroundColor: dataTB.map(val => val > 120 ? "#FF4B4B" : "#00F2FE"),
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94A3B8" }, grid: { display: false } },
        y: { ticks: { color: "#94A3B8" }, grid: { color: "rgba(255,255,255,0.05)" } }
      }
    }
  });

  const listContainer = document.getElementById("circle-capacity-list");
  listContainer.innerHTML = "";

  circles.forEach(c => {
    const div = document.createElement("div");
    div.style.background = "rgba(0,0,0,0.2)";
    div.style.border = "1px solid rgba(255,255,255,0.06)";
    div.style.padding = "10px 14px";
    div.style.borderRadius = "10px";
    div.style.display = "flex";
    div.style.justifyContent = "space-between";
    div.style.alignItems = "center";

    const isHigh = c.projected_monthly_tb > 120;
    div.innerHTML = `
      <div>
        <strong style="color: #FFF;">${c.circle}</strong>
        <div style="font-size: 11px; color: #94A3B8;">${c.current_daily_gb} GB/day current baseline</div>
      </div>
      <div style="text-align: right;">
        <span style="font-weight: 700; color: ${isHigh ? '#FF4B4B' : '#00F2A9'};">${c.projected_monthly_tb} TB/mo</span>
        <div style="font-size: 11px; color: ${isHigh ? '#FF4B4B' : '#94A3B8'};">${c.capacity_status}</div>
      </div>
    `;
    listContainer.appendChild(div);
  });
}

function setupEventListeners() {
  // Boost slider interaction
  const boostSlider = document.getElementById("sim-boost-slider");
  boostSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    document.getElementById("sim-boost-val").textContent = `+${val}%`;
    renderGrowthForecastChart(parseInt(val));
  });

  // Horizon slider interaction
  const horizonSlider = document.getElementById("sim-horizon-slider");
  horizonSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    document.getElementById("sim-horizon-val").textContent = `${val} Days`;
    const boostVal = parseInt(document.getElementById("sim-boost-slider").value);
    renderGrowthForecastChart(boostVal);
  });

  // Search and filter churn table
  document.getElementById("search-customer").addEventListener("input", renderChurnTable);
  document.getElementById("filter-risk").addEventListener("change", renderChurnTable);

  // NBO Customer Lookup
  document.getElementById("nbo-lookup-btn").addEventListener("click", () => {
    const id = document.getElementById("nbo-search-id").value.trim();
    if (id) inspectCustomerNBO(id);
  });

  // Modal interactions
  document.getElementById("modal-close-btn").addEventListener("click", closeModal);
  document.getElementById("modal-cancel-btn").addEventListener("click", closeModal);
  document.getElementById("modal-confirm-btn").addEventListener("click", () => {
    alert("🚀 Retention offer dispatched successfully via WhatsApp & SMS!");
    closeModal();
  });
}

function inspectCustomerNBO(customerId) {
  const customer = mlData.customers.find(c => c.customer_id.toLowerCase() === customerId.toLowerCase());
  const display = document.getElementById("nbo-profile-display");

  if (!customer) {
    alert("Customer ID not found. Please try another ID (e.g. C000001 to C002000).");
    return;
  }

  display.style.display = "block";
  document.getElementById("nbo-customer-name").textContent = `${customer.name} (${customer.customer_id})`;
  
  const riskBadge = document.getElementById("nbo-risk-badge");
  riskBadge.className = `risk-badge risk-${customer.risk_level}`;
  riskBadge.textContent = `${customer.risk_level} Churn Risk (${customer.churn_probability}%)`;

  document.getElementById("nbo-current-plan").textContent = customer.current_plan;
  document.getElementById("nbo-pred-recharge").textContent = `₹${customer.predicted_next_recharge}`;
  document.getElementById("nbo-rfm-segment").textContent = customer.rfm_segment;
  document.getElementById("nbo-recency").textContent = `${customer.recency_days} days ago`;

  document.getElementById("nbo-rec-plan").textContent = customer.nbo_plan_name;
  document.getElementById("nbo-rec-reason").textContent = `Rationale: ${customer.nbo_reason}`;
}

function openInterventionModal(id, name, prob) {
  document.getElementById("modal-title").textContent = `Churn Intervention: ${name} (${id})`;
  document.getElementById("modal-body").innerHTML = `
    <p style="margin-bottom: 12px; font-size: 14px;">This customer has a <strong>${prob}% predicted churn probability</strong>.</p>
    <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 14px; border-radius: 10px;">
      <label style="display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">Select Discount Campaign Offer:</label>
      <select class="custom-select" style="width: 100%;">
        <option>20% Win-Back Discount on Truly Unlimited (₹479 → ₹239)</option>
        <option>Free 15GB Data Booster Coupon (₹148 Value)</option>
        <option>Direct Dedicated Account Executive Outreach</option>
      </select>
    </div>
  `;
  document.getElementById("action-modal").classList.add("active");
}

function closeModal() {
  document.getElementById("action-modal").classList.remove("active");
}
