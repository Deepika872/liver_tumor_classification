
import { Link, NavLink } from "react-router-dom";
import { Activity } from "lucide-react";
import appConfig from "../config/appConfig";

function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar__inner">
        <Link to="/" className="navbar__brand">
          <span className="navbar__logo">
            <Activity size={23} />
          </span>

          <span className="navbar__brand-text">
            {appConfig.shortName}
            <small>AI Research Project</small>
          </span>
        </Link>

        <nav className="navbar__links" aria-label="Main navigation">
          <NavLink to="/" end>
            Home
          </NavLink>

          <NavLink to="/upload">
            Upload Image
          </NavLink>

          <NavLink to="/about">
            About
          </NavLink>
        </nav>

        <Link to="/upload" className="navbar__cta">
          Get Started
        </Link>
      </div>
    </header>
  );
}

export default Navbar;
