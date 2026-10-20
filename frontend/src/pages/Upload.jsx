
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload as UploadIcon,
  Image as ImageIcon,
  X,
  ShieldCheck,
  FileImage,
} from "lucide-react";

import Button from "../components/Button";
import appConfig from "../config/appConfig";
import { classifyImage } from "../services/predictionApi";
import "./Upload.css";

function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const inputRef = useRef(null);
  const navigate = useNavigate();

  function handleFile(file) {
    setError("");

    if (!file) return;

    if (!appConfig.allowedImageTypes.includes(file.type)) {
      setError("Please select a JPG or PNG image.");
      return;
    }

    const maxSize = appConfig.maxUploadSizeMB * 1024 * 1024;

    if (file.size > maxSize) {
      setError(
        `Image size must not exceed ${appConfig.maxUploadSizeMB} MB.`
      );
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  }

  function handleInputChange(event) {
    handleFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function removeImage() {
    if (preview) URL.revokeObjectURL(preview);

    setSelectedFile(null);
    setPreview("");
    setError("");
  }

  async function handleContinue() {
    if (!selectedFile) {
      setError("Please select an image before continuing.");
      return;
    }

    setError("");
    setIsLoading(true);

    try {
      const prediction = await classifyImage(selectedFile);

      navigate("/result", {
        state: {
          imagePreview: preview,
          fileName: selectedFile.name,
          prediction,
        },
      });
    } catch (err) {
      console.error("Image upload failed:", err);

      setError(
        err.response?.data?.detail ||
          "Unable to connect to the backend. Please check that the Python server is running."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="upload-page">
      <div className="upload-heading">
        <span className="upload-eyebrow">
          <ImageIcon size={16} /> IMAGE CLASSIFICATION
        </span>

        <h1>Upload Your Image</h1>

        <p>
          Select a liver image to begin the image classification
          workflow using our planned {appConfig.modelName} model.
        </p>
      </div>

      <div className="upload-card">
        {!preview ? (
          <div
            className={`upload-dropzone ${
              isDragging ? "upload-dropzone--active" : ""
            }`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              handleFile(event.dataTransfer.files?.[0]);
            }}
          >
            <div className="upload-icon">
              <UploadIcon size={30} />
            </div>

            <h2>Choose an image to upload</h2>
            <p>Drag and drop your image here, or browse your computer.</p>

            <Button onClick={() => inputRef.current?.click()}>
              <FileImage size={18} /> Browse Images
            </Button>

            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleInputChange}
              hidden
            />

            <span className="upload-file-info">
              Supported formats: JPG, PNG · Maximum size:{" "}
              {appConfig.maxUploadSizeMB} MB
            </span>
          </div>
        ) : (
          <div className="upload-preview-section">
            <div className="upload-preview-header">
              <div>
                <h2>Image Preview</h2>
                <p>Check your selected image before continuing.</p>
              </div>

              <button
                type="button"
                className="upload-remove-button"
                onClick={removeImage}
                aria-label="Remove selected image"
                disabled={isLoading}
              >
                <X size={20} />
              </button>
            </div>

            <div className="upload-preview-image-wrapper">
              <img
                src={preview}
                alt="Selected image preview"
                className="upload-preview-image"
              />
            </div>

            <p className="upload-file-name">
              <FileImage size={17} />
              {selectedFile.name}
            </p>

            <div className="upload-preview-actions">
              <Button
                variant="secondary"
                onClick={() => inputRef.current?.click()}
                disabled={isLoading}
              >
                Choose Another
              </Button>

              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleInputChange}
                hidden
              />

              <Button
                onClick={handleContinue}
                disabled={isLoading}
              >
                {isLoading ? "Uploading..." : "Send Image for Analysis"}
              </Button>
            </div>
          </div>
        )}

        {error && (
          <p className="upload-error" role="alert">
            {error}
          </p>
        )}

        <div className="upload-privacy-note">
          <ShieldCheck size={19} />
          <p>
            <strong>Research use only.</strong> The backend currently
            validates uploaded images. Tumor prediction will be available
            after the trained model is integrated.
          </p>
        </div>
      </div>
    </section>
  );
}

export default Upload;
