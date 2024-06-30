const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const imgView = document.getElementById("img-view");
const submitButton = document.getElementById("submit-button");
const resetButton = document.getElementById("reset-button");
const predictionResult = document.getElementById("prediction-result");

inputFile.addEventListener("change", uploadImage);

function uploadImage() {
  let imgLink = URL.createObjectURL(inputFile.files[0]);
  imgView.style.backgroundImage = `url(${imgLink})`;
  imgView.textContent = "";
  imgView.style.border = 0;
  imgView.style.backgroundColor = "black";
}

dropArea.addEventListener("dragover", function (e) {
  e.preventDefault();
});

dropArea.addEventListener("drop", function (e) {
  e.preventDefault();
  inputFile.files = e.dataTransfer.files;
  uploadImage();
});

submitButton.addEventListener("click", makePrediction);
resetButton.addEventListener("click", resetAll);

function resetAll() {
  inputFile.value = "";
  imgView.innerHTML = `
    <img src="../static/upload.png" style="max-width: 100%; max-height: 100%;"/>
    <p>Drag and drop or click to upload an image</p>
    <span>Upload an image from desktop</span>
    <span></span>
  `;
  imgView.style.border = "2px dashed #ccc";
  imgView.style.backgroundColor = "transparent";
  imgView.style.backgroundImage = "none";
  predictionResult.textContent = "";
  submitButton.style.display = "block";
  resetButton.style.display = "none";
}

async function makePrediction() {
  if (!inputFile.files[0]) {
    alert("Please, select an image.");
    return;
  }
  const formData = new FormData();
  formData.append("file", inputFile.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });
    submitButton.style.display = "none";
    const result = await response.json();
    predictionResult.textContent = result.prediction;
    resetButton.style.display = "block";
  } catch (error) {
    console.log("Error:", error);
    alert("Request Failed");
  }
}
