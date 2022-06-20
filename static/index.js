function uploadFiles() {
  const root = document.getElementById("form");
  root.submit();
}

function addUploadForm() {
  const root = document.getElementById("form");
  let childCount = (root.childElementCount + 1).toString();
  const form = document.getElementById("1");
  const newForm = form.cloneNode();

  newForm.id = childCount;
  newForm.innerHTML =
    '<div class="mb-2"><input type="file" name="file-' +
    childCount +
    '" multiple="" id="files" /></div><div class="mb-2">' +
    '<input type="text" placeholder="class ' +
    childCount +
    '" name="class_name-' +
    childCount +
    '" id="class_name" required /> </div>' +
    '<div class="mb-2"><input type="radio" name="file_type-' +
    childCount +
    '" id="image" value="image" checked />' +
    '<label for="image">&nbsp;Image&nbsp;</label><input type="radio" name="file_type-' +
    childCount +
    '" id="video" value="video" />' +
    '<label for="video">&nbsp;Video</label></div>';

  root.appendChild(newForm);
}
