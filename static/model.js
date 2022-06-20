function showTuning() {
  ele = document.getElementById("tuning");
  if (ele.hidden) ele.hidden = false;
  else ele.hidden = true;
}

function train() {
  btn = document.getElementById("train");
  btn.disabled = true;
  document.getElementById("status").hidden = false;
  form = document.getElementById("form");
  form.submit();
}
