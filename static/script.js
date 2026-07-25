// Coronary Heart Disease Prediction - Frontend JavaScript

document.addEventListener("DOMContentLoaded", function () {
  // Form validation and enhancement
  const form = document.getElementById("predictionForm");

  if (form) {
    // Add input validation
    const inputs = form.querySelectorAll("input, select");
    inputs.forEach((input) => {
      input.addEventListener("change", function () {
        validateInput(this);
      });

      input.addEventListener("blur", function () {
        validateInput(this);
      });
    });

    // Form submission with loading state
    form.addEventListener("submit", function (e) {
      const isValid = validateForm();
      if (!isValid) {
        e.preventDefault();
        return;
      }

      // Show loading state
      const submitBtn = form.querySelector('button[type="submit"]');
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner"></span> Predicting...';
    });
  }

  // Cigarettes per day auto-update based on smoking status
  const smokerSelect = document.getElementById("currentSmoker");
  const cigsInput = document.getElementById("cigsPerDay");

  if (smokerSelect && cigsInput) {
    smokerSelect.addEventListener("change", function () {
      if (this.value === "0") {
        cigsInput.value = "0";
        cigsInput.disabled = true;
      } else {
        cigsInput.disabled = false;
        cigsInput.value = "";
        cigsInput.focus();
      }
    });

    // Initialize state
    if (smokerSelect.value === "0") {
      cigsInput.disabled = true;
    }
  }

  // BMI calculator hint
  const heightInput = document.getElementById("height");
  const weightInput = document.getElementById("weight");
  const bmiInput = document.getElementById("BMI");

  // Animate result on result page
  const riskScore = document.querySelector(".risk-score .percentage");
  if (riskScore) {
    animateValue(riskScore, 0, parseFloat(riskScore.textContent), 1500);
  }
});

// Validate individual input
function validateInput(input) {
  const value = parseFloat(input.value);
  const min = parseFloat(input.min);
  const max = parseFloat(input.max);

  if (!isNaN(value) && !isNaN(min) && !isNaN(max)) {
    if (value < min || value > max) {
      input.style.borderColor = "#ef4444";
      return false;
    }
  }

  input.style.borderColor = "#e2e8f0";
  return true;
}

// Validate entire form
function validateForm() {
  const form = document.getElementById("predictionForm");
  const inputs = form.querySelectorAll("input[required], select[required]");
  let isValid = true;

  inputs.forEach((input) => {
    if (!input.value || input.value.trim() === "") {
      input.style.borderColor = "#ef4444";
      isValid = false;
    } else if (!validateInput(input)) {
      isValid = false;
    }
  });

  if (!isValid) {
    alert("Please fill in all required fields with valid values.");
  }

  return isValid;
}

// Animate number value
function animateValue(element, start, end, duration) {
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // Easing function
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * easeOut;

    element.textContent = current.toFixed(1) + "%";

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute("href"));
    if (target) {
      target.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  });
});
