# FabricFlow Shared Library

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](VERSION)
[![CI Status](https://github.com/hpi-main/solution-fabric-shared-library/workflows/CI/badge.svg)](https://github.com/hpi-main/solution-fabric-shared-library/actions)

> **Official repository of reusable FabricFlow tasks and workflows**

A version-controlled, community-driven collection of production-ready tasks and workflows for the FabricFlow ecosystem. Consumed by FabricFlow Studio, Compute Fabric, and the FabricFlow CLI.

## 🌟 What's Inside

- **115+ Tasks**: Reusable operations across 17 categories (image processing, ML inference, LLMs, document processing, etc.)
- **9 Workflows**: Complete end-to-end pipelines (object detection, document scanning, stable diffusion, etc.)
- **Metadata-Driven**: Rich metadata for discovery and automated deployment
- **Production-Ready**: Thoroughly tested, documented, and validated
- **Community-Powered**: Open for contributions from the FabricFlow community

### Using FabricFlow Studio

1. Open FabricFlow Studio desktop application
2. Navigate to **Task Library** panel
3. Browse tasks by category
4. Drag and drop tasks onto the canvas
5. Connect tasks and configure inputs
6. Run your workflow

## 📂 Repository Structure

```
solution-fabric-shared-library/
├── tasks/                  # Task implementations organized by category
│   ├── {category}/        # Category folders (array, audio, camera, diffusion, etc.)
│   │   └── *.py           # Task implementation files
│   └── _metadata/         # Auto-generated task metadata JSON files
│       └── {category}/    # Metadata organized by category
│
├── workflows/              # Workflow JSON definitions
│   ├── *.json             # Workflow files (e.g., yolo_object_detection.json)
│   └── _metadata/         # Auto-generated workflow metadata
│
├── samples/                # Example code and media files
│   ├── api/               # REST API client examples
│   ├── sdk/               # Python SDK examples
│   ├── streamlit/         # Streamlit app examples
│   ├── media/             # Sample images and videos for workflows
│   └── README.md
│
├── scripts/                # Validation and generation tools
│   ├── generate_manifest.py
│   ├── validate_workflow.py
│   └── README.md
│
├── .github/                # GitHub configuration and CI/CD
│
├── VERSION                 # Repository version
├── CHANGELOG.md           # Version history
└── README.md              # This file
```

## 🎯 Featured Workflows

### Computer Vision

- **YOLO Object Detection**: Detect objects in images with bounding boxes
- **Streaming Webcam Detection**: Real-time object detection from webcam
- **Document Scanning**: OCR and text extraction from documents

### Generative AI

- **Stable Diffusion**: Generate images from text prompts
- **LLaMA Fine-tuning**: Fine-tune LLaMA models on custom datasets
- **Grammar Helper**: AI-powered grammar correction and improvement

### Image Processing

- **Camera Background Blur**: Apply background blur to webcam feed
- **Image Enhancement Pipeline**: Resize, normalize, and enhance images

## 🤝 Contributing

We welcome contributions from the community! Whether you're adding new tasks, workflows, or improving existing ones.

### Quick Contribution Guide

1. **Fork and clone** the repository
2. **Create a branch**: `git checkout -b feature/my-new-task`
3. **Add your task/workflow**: Follow the task development patterns in existing tasks
4. **Validate**: Run `python scripts/generate_manifest.py` to validate and generate metadata
5. **Test**: Ensure your task works with sample workflows
6. **Submit PR**: Open a pull request with clear description

### What Can You Contribute?

- ✨ **New Tasks**: Reusable operations in any domain
- 🔄 **New Workflows**: Complete pipelines solving real problems
- 🐛 **Bug Fixes**: Improvements to existing tasks/workflows
- 📚 **Documentation**: Better guides, examples, and explanations
- 🏷️ **Metadata**: Enhanced descriptions and tags

## 📖 Documentation

- **[Scripts README](scripts/README.md)**: Validation and metadata generation tools
- **[Samples README](samples/README.md)**: Example code and media files
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and changes

## 🔧 Requirements

- **FabricFlow**: Version 0.6.0 or higher
- **Python**: 3.12 or higher
- **Git**: For syncing the library

Task-specific requirements vary. Check individual task metadata for dependencies.

## 📊 Statistics

- **Total Tasks**: 115+
- **Total Workflows**: 9
- **Categories**: 17
- **Contributors**: [See Contributors](https://github.com/hpi-main/solution-fabric-shared-library/graphs/contributors)
- **Version**: 1.0.0

## 🔗 Related Projects

- **[FabricFlow](https://github.com/hpi-main/solution-fabric-flow)**: Core workflow engine and CLI
- **[FabricFlow Studio](https://github.com/hpi-main/solution-fabric-studio)**: Desktop application for visual workflow design
- **[Compute Fabric](https://github.com/hpi-main/solution-compute-fabric)**: Enterprise workflow orchestration platform

## 📄 License

See repository license for details.

## 💬 Support

- **Issues**: [Report bugs or request features](https://github.com/hpi-main/solution-fabric-shared-library/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/hpi-main/solution-fabric-shared-library/discussions)
- **Maintainers**: [@hpi-main/solution-fabric-shared-maintainers](https://github.com/orgs/hpi-main/teams/solution-fabric-shared-maintainers)

## 🚀 Roadmap

- [ ] 200+ tasks across all categories
- [ ] 25+ production workflows
- [ ] Enhanced metadata with usage examples
- [ ] Automated workflow testing
- [ ] Performance benchmarking
- [ ] Task usage analytics

---

## Ownership

This repository is maintained by:
- **Admins:** [@hpi-main/solution-fabric-shared-admins](https://github.com/orgs/hpi-main/teams/solution-fabric-shared-admins)
- **Maintainers:** [@hpi-main/solution-fabric-shared-maintainers](https://github.com/orgs/hpi-main/teams/solution-fabric-shared-maintainers)

---

**Made with ❤️ by the FabricFlow Community**
