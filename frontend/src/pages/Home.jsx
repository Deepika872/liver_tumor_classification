
import { Link } from "react-router-dom";
import { Activity, ShieldCheck, Image as ImageIcon } from "lucide-react";
import appConfig from "../config/appConfig";
import Button from "../components/Button";

function Home() {
  return (
    <section className="home-page">
      <div className="home-hero">
        <div className="home-content">
          <p className="eyebrow">
            <Activity size={16} /> AI-POWERED RESEARCH
          </p>

          <h1>
            Smarter Insights for{" "}
            <span>Liver Image Classification</span>
          </h1>

          <p className="home-description">
            Explore a deep learning research prototype designed to study
            liver tumor image classification using {appConfig.modelName}.
          </p>

          <div className="home-actions">
            <Link to="/upload">
              <Button>
                <ImageIcon size={18} /> Upload Image
              </Button>
            </Link>

            <Link to="/about">
              <Button variant="secondary">Explore Project</Button>
            </Link>
          </div>

          <div className="home-note">
            <ShieldCheck size={18} />
            <span>Academic research prototype — not for clinical diagnosis.</span>
          </div>
        </div>

        <div className="home-visual">
          <div className="home-visual-icon">
            <Activity size={76} strokeWidth={1.3} />
          </div>
          <p className="home-visual-title">{appConfig.shortName}</p>
          <p className="home-visual-subtitle">
            Deep Learning · Image Classification
          </p>

          <div className="home-model-card">
            <span>Planned model architecture</span>
            <strong>{appConfig.modelName}</strong>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Home;
