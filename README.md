<br>
<img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320">

# 🛠 Ultralytics Python Project Template

This repository serves as the template for Python projects at [Ultralytics](https://ultralytics.com). It encapsulates best practices, standard configurations, and essential project structures, streamlining the initiation process for new Python projects. By leveraging this template, developers at Ultralytics can ensure consistency and adherence to quality standards across all Python-based software developments.

[![Ultralytics Actions](https://github.com/ultralytics/template/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/template/actions/workflows/format.yml)

## 🗂 Repository Structure

The repository is meticulously organized to offer intuitive navigation and a clear understanding of the project components:

- `src/` or `your_package_name/`: Contains the source code of the Python package, organized in modules and packages.
- `tests/`: Dedicated to unit tests and integration tests, facilitating continuous testing practices.
- `docs/`: (Optional) Houses project documentation, typically managed with tools like Sphinx.
- `requirements.txt` or `Pipfile`: Lists all necessary Python package dependencies.
- `.gitignore`: Configured to exclude unnecessary files from Git tracking.
- `LICENSE`: Specifies the open-source license under which the project is released.
- `.github/workflows/`: Contains GitHub Actions workflows for CI/CD processes.
- `.pre-commit-config.yaml`: (Optional) Pre-commit hooks configuration for maintaining code quality.
- `Dockerfile`: (Optional) For containerizing the project environment.
- `environment.yml`: (Optional, for Conda users) Manages Conda environment dependencies.
- `setup.py`: (Optional, if using PyPI) Details for packaging and distributing the project.
- Linting and formatting configuration files (like `.flake8`, `.pylintrc`, `pyproject.toml`).

```
your-project/
│
├── your_package_name/
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   └── ...
│
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── ...
│
├── docs/
│   └── ...
│
├── pyproject.toml
└── README.md
```

### Source Code in `src/` or `your_package_name/` Directory 📂

The `src/` or `your_package_name/` directory is the heart of your project, containing the Python code that constitutes your package. This structure encourages clean imports and testing practices.

### Testing with the `tests/` Directory 🧪

The `tests/` directory is crucial for maintaining the reliability and robustness of your code. It should include comprehensive tests that cover various aspects of your package.

### Documentation in `docs/` Directory 📚

For projects requiring extensive documentation, the `docs/` directory serves as the go-to place. It's typically set up with Sphinx for generating high-quality documentation.

## ➕ Starting a New Project

To kickstart a new Python project with this template:

1. **Create Your New Repository**: Use this template to generate a new repository for your project.
2. **Customize the Template**: Tailor the template files like `requirements.txt`, `.pre-commit-config.yaml`, and GitHub workflow YAMLs to suit your project's needs.
3. **Develop Your Package**: Begin adding your code into the `src/` or `your_package_name/` directory and corresponding tests in the `tests/` directory.
4. **Document Your Project**: Update the README and, if necessary, add documentation to the `docs/` directory.
5. **Continuous Integration**: Leverage the pre-configured GitHub Actions for automated testing and other CI/CD processes.

## 🔧 Utilizing the Template

For Ultralytics team members and contributors:

- Clone the template repository to get started on a new Python project.
- Update the `README.md` to reflect your project's specifics.
- Remove or modify any optional components (like `Dockerfile`, `environment.yml`) based on the project's requirements.

With this template, Ultralytics aims to foster a culture of excellence and uniformity in Python software development, ensuring that each project is built on a solid foundation of industry standards and organizational best practices.

## 💡 Contribute

Ultralytics thrives on community collaboration; we immensely value your involvement! We urge you to peruse our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for detailed insights on how you can participate. Don't forget to share your feedback with us by contributing to our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 goes out to everyone who has already contributed!

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## 📄 License

Ultralytics presents two distinct licensing paths to accommodate a variety of scenarios:

- **AGPL-3.0 License**: This official [OSI-approved](https://opensource.org/licenses/) open-source license is perfectly aligned with the goals of students, enthusiasts, and researchers who believe in the virtues of open collaboration and shared wisdom. Details are available in the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) document.
- **Enterprise License**: Tailored for commercial deployment, this license authorizes the unfettered integration of Ultralytics software and AI models within commercial goods and services, without the copyleft stipulations of AGPL-3.0. Should your use case demand an enterprise solution, direct your inquiries to [Ultralytics Licensing](https://ultralytics.com/license).

## 📮 Contact

For bugs or feature suggestions pertaining to Ultralytics, please lodge an issue via [GitHub Issues](https://github.com/ultralytics/pre-commit/issues). You're also invited to participate in our [Discord](https://ultralytics.com/discord) community to engage in discussions and seek advice!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
