let interval = setInterval(updateCountdown, 500);

const result = (values) => {
  clearInterval(interval);

  const wrapper = document.createElement("div");
  wrapper.innerHTML = [
    '<div id="result" class="text-center">',
    `   <p class="fs-2 vw-100 text-dark">${values}</p>`,
    '   <a href="/dc" class="fs-3 pt-3 text-dark">Restart</a>',
    "</div>",
  ].join("");
  document.body.style.backgroundColor = "var(--bs-info)";
  let countdownAlert = document.getElementById("countdown");
  countdownAlert.replaceWith(wrapper);
};

const go = () => {
  clearInterval(interval);

  const wrapper = document.createElement("div");
  wrapper.innerHTML = [
    '<div id="countdown">',
    '   <p class="fs-1 text-dark">THROW</p>',
    "</div>",
  ].join("");
  document.body.style.backgroundColor = "var(--bs-success)";
  let countdownAlert = document.getElementById("countdown");
  countdownAlert.replaceWith(wrapper);

  interval = setInterval(showResults, 5000);
};

const countdown = (message, type) => {
  const wrapper = document.createElement("div");
  wrapper.innerHTML = [
    `<div id="countdown">`,
    `   <p class="fs-1 text-dark">${message}</p>`,
    "</div>",
  ].join("");
  document.body.style.backgroundColor = `var(--bs-${type})`;
  let countdownAlert = document.getElementById("countdown");
  countdownAlert.replaceWith(wrapper);
};

async function updateCountdown() {
  let update = await fetch("/dc/countdown");
  if (update.status === 200) {
    update = await update.json();
    switch (update.VALUE) {
      case "0":
        go();
        break;
      case "1":
        countdown("1", "warning");
        break;
      default:
        countdown(update.VALUE, "danger");
    }
  }
}

async function showResults() {
  let update = await fetch("/dc/gvalue");
  if (update.status === 200) {
    update = await update.json();
    result(update.VALUE);
  }
}
