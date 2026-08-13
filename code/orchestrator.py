"""ValenceSphere v11 application entry point."""
# https://github.com/Kosev-Lex
# Valence Sphere - experimental concept first AI reasoning system, using triadic questioner (Socrates), answerer (Plato), and adjudicator
# for data auditing and verification to give AI models greater structure and the beginnings of critical thinking.
# Publicly released open source on 14 August 2026. Concept first created 30 July 2025.
# made by JL Kosev-Lex

from gui import ValenceSphereGUI


def main():
    app = ValenceSphereGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
