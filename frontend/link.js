async function sendToBackend(files) {
  const formData = new FormData();

  for (let file of files) {
    formData.append("files", file);
  }

  const response = await fetch("http://localhost:5000/api/upload", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  return data;
}
