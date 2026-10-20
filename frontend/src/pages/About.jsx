
import appConfig from "../config/appConfig";

function About() {
  return (
    <section className="page-container">
      <div className="about-page">
        <p className="eyebrow">ABOUT THE PROJECT</p>

        <h1>
          About {appConfig.name}
        </h1>

        <p>
          This project explores the use of deep learning to classify liver
          tumor images into benign and malignant categories.
        </p>

        <h2>Our Approach</h2>

        <p>
          We plan to use the {appConfig.modelName} architecture for image
          classification, with image preprocessing and model evaluation.
        </p>

        <h2>Project Purpose</h2>

        <p>
          This is an academic research prototype designed for learning and
          research. It is not intended for clinical diagnosis or treatment
          decisions.
        </p>
      </div>
    </section>
  );
}

export default About;
