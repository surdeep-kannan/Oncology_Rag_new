import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./features/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./services/**/*.{ts,tsx}",
    "./store/**/*.{ts,tsx}",
    "./utils/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  darkMode: ["class"],
  theme: {
    extend: {
      colors: {
        canvas: {
          50: "#f6f7f5",
          100: "#ecefe8",
          900: "#0f1417",
          950: "#091016"
        },
        teal: {
          300: "#7be4da",
          400: "#48d2c3",
          500: "#1db4a8",
          600: "#11857f"
        },
        coral: {
          300: "#ffc5b6",
          400: "#ff9d82",
          500: "#ff7a59"
        }
      },
      boxShadow: {
        glow: "0 18px 60px rgba(16, 185, 129, 0.15)"
      },
      backgroundImage: {
        "mesh-light":
          "radial-gradient(circle at top left, rgba(29,180,168,0.16), transparent 30%), radial-gradient(circle at 85% 15%, rgba(255,122,89,0.14), transparent 24%), linear-gradient(180deg, #f9fbfa 0%, #edf2ef 100%)",
        "mesh-dark":
          "radial-gradient(circle at top left, rgba(72,210,195,0.18), transparent 28%), radial-gradient(circle at 85% 15%, rgba(255,122,89,0.12), transparent 22%), linear-gradient(180deg, #071015 0%, #0f1720 100%)"
      }
    }
  },
  plugins: []
};

export default config;
