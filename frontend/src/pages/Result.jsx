
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowLeft,
  Image as ImageIcon,
  RotateCcw,
  ShieldAlert,
} from "lucide-react";
import Button from "../components/Button";
import "./Result.css";

function Result() {
  const location = useLocation();
  const navigate = useNavigate();

  const imagePreview = location.state?.imagePreview;
  const prediction = location.state?.prediction;

  if (!imagePreview) {
    return (
      <section className="result-page">
        <div className="result-empty">
          <div className="result-icon">
            <ImageIcon size={32} />
          </div>

          <h1>No Image Available</h1>

          <p>
            Please upload an image first to view the classification result.
          </p>

          <Button onClick={() => navigate("/upload")}>
            <ArrowLeft size={18} />
            Go to Upload
          </Button>
        </div>
      </section>
    );
  }

  return (
    <section className="result-page">
      <div className="result-heading">
        <p className="eyebrow">
          <Activity size={16} />
          IMAGE ANALYSIS
        </p>

        <h1>Classification Result</h1>

        <p>
          Review your uploaded image and the prediction returned by the model.
        </p>
      </div>

      <div className="result-grid">
        <div className="result-card">
          <h2>Uploaded Image</h2>

          <div className="result-image-wrapper">
            <img
              src={imagePreview}
              alt="Uploaded liver scan preview"
              className="result-image"
            />
          </div>

          <p className="result-caption">
            Your selected image
          </p>

          <Button
            variant="secondary"
            onClick={() => navigate("/upload")}
          >
            <RotateCcw size={17} />
            Upload Another Image
          </Button>
        </div>

        <div className="result-card result-details">
          <h2>Analysis Summary</h2>

          {prediction ? (
            <>
              <div className="prediction-label">
                <span>Predicted Class</span>
                <strong>{prediction.class_name ?? prediction.prediction ?? "Result unavailable"}</strong>
              </div>

              {prediction.confidence != null && (
                <div className="confidence-section">
                  <div className="confidence-header">
                    <span>Model Confidence</span>
                    <strong>
                      {Number(prediction.confidence) <= 1
                        ? `${(Number(prediction.confidence) * 100).toFixed(1)}%`
                        : `${Number(prediction.confidence).toFixed(1)}%`}
                    </strong>
                  </div>

                  <div className="confidence-track">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${
                          Number(prediction.confidence) <= 1
                            ? Math.max(0, Math.min(100, Number(prediction.confidence) * 100))
                            : Math.max(0, Math.min(100, Number(prediction.confidence)))
                        }%`,
                      }}
                    />
                  </div>
                </div>
              )}

              <p className="result-message">
                The displayed prediction is provided by the connected model.
                Please consult a qualified medical professional for medical
                interpretation.
              </p>
            </>
          ) : (
            <div className="result-pending">
              <div className="pending-icon">
                <Activity size={25} />
              </div>

              <h3>Prediction Not Available Yet</h3>

              <p>
                Your image preview is available, but the EfficientNetV2
                prediction API has not been connected to this page yet.
              </p>

              <span className="pending-status">
                Waiting for model integration
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="result-disclaimer">
        <ShieldAlert size={20} />
        <p>
          <strong>Research prototype only:</strong> This application is not
          intended for clinical diagnosis or treatment decisions.
        </p>
      </div>

      <Link to="/" className="result-home-link">
        <ArrowLeft size={16} />
        Back to Home
      </Link>
    </section>
  );
}

export default Result;
