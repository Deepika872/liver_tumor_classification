
import appConfig from "../config/appConfig";

function Footer() {
  return (
    <footer className="footer">
      <div className="footer__inner">
        <p>
          © {new Date().getFullYear()} {appConfig.name}
        </p>

        <p className="footer__note">
          Academic research prototype — not for clinical diagnosis.
        </p>
      </div>
    </footer>
  );
}

export default Footer;
