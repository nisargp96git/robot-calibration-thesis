# Development Environment Setup

## Purpose

This document records the development environment used for the master's thesis project. Commands and configuration changes are documented together with their purpose so that the environment can be reproduced and understood later.

## Operating System

- Ubuntu 24.04
- Project directory: `~/Documents/Thesis`

## Development Tools

### Visual Studio Code

Visual Studio Code is used as the primary editor for source code and project documentation.

Verification command:

```bash
code --version

Purpose:
Checks whether the Visual Studio Code command-line interface is installed and accessible from the terminal.


Git
Git is used for local version control.
Verification command:
git --version
Purpose:
Displays the installed Git version and confirms that Git is available.

The repository was initialized using:
git init
Purpose:
Creates a Git repository in the current directory by creating the hidden .git/ directory.

The primary branch was renamed to main:
git branch -m main
The default branch name for future repositories was configured using:
git config --global init.defaultBranch main

GitHub
GitHub is used as the remote repository host.
SSH authentication is used between the development computer and GitHub. A dedicated Ed25519 SSH key was created for GitHub authentication.
The private SSH key is stored locally and must never be committed to the repository or shared.
SSH authentication was verified using:
ssh -T git@github.com

The local Git repository is connected to the GitHub repository using a remote named origin.
Git Ignore Rules
The .gitignore file prevents generated, temporary, environment-specific, and selected large data files from being accidentally committed.
Examples currently ignored include:
- Python virtual environments
- Python cache files
- VS Code workspace settings
- temporary files
- data/raw/
- selected SolidWorks temporary/backup files
Documentation Principle
Commands, configuration changes, experiments, problems, and solutions should be documented throughout the project so that the thesis workflow remains reproducible.


### Why are we doing this now?

Your Git repository tells us **what changed**, but Git itself does not adequately explain **why you made an engineering decision**.

For example:

```bash
git config --global init.defaultBranch main

may appear in your terminal history, but six months later that doesn't explain why it was used.
This documentation gives us two complementary records:
Git history
    → What changed and when

Engineering documentation
    → What was done, how, and why

That distinction becomes much more important when we reach camera calibration, robot poses, coordinate frames, calibration parameters, experimental conditions, and error analysis.