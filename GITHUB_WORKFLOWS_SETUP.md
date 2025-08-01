# 🚀 GitHub Actions Workflow Setup

## Overview

Due to GitHub's security permissions, automated tools cannot directly create or modify workflow files in `.github/workflows/`. This document provides instructions for manually setting up the comprehensive CI/CD pipeline.

## 📋 Setup Instructions

### 1. Copy the CI/CD Template

Copy the pre-configured workflow template to activate the CI/CD pipeline:

```bash
# Copy the template to activate CI/CD
cp docs/workflows/ci-template.yml .github/workflows/ci.yml

# Commit the workflow
git add .github/workflows/ci.yml
git commit -m "ci: activate comprehensive CI/CD pipeline"
git push
```

### 2. Configure Repository Secrets

Add these secrets in your GitHub repository settings (Settings → Secrets and variables → Actions):

#### Required Secrets:
- `PYPI_API_TOKEN`: For automated package publishing
- `CODECOV_TOKEN`: For coverage reporting (optional)

#### Optional Secrets for Enhanced Features:
- `OPENAI_TEST_API_KEY`: For testing LLM integrations
- `ANTHROPIC_TEST_API_KEY`: For testing AI features

### 3. Workflow Features

The CI/CD pipeline includes:

#### 🔧 **Code Quality & Security**
- **Formatting**: Black, isort
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: MyPy with strict configuration
- **Security Scanning**: Bandit, Safety, pip-audit

#### 🧪 **Comprehensive Testing**
- **Multi-platform**: Ubuntu, macOS
- **Multi-version**: Python 3.9, 3.10, 3.11, 3.12
- **Test Types**: Unit, integration, performance
- **Coverage**: Automatic coverage reporting

#### 📦 **Build & Package**
- **Package Building**: Automated wheel and source distribution
- **Installation Testing**: Validates package installation
- **Docker**: Multi-stage container builds

#### 🚀 **Release Automation**
- **Automated Releases**: Tag-triggered PyPI publishing
- **GitHub Releases**: Automatic release notes generation
- **Artifact Management**: Build artifact preservation

#### 📊 **Advanced Features**
- **Dependency Updates**: Scheduled security scans
- **Performance Monitoring**: Benchmark tracking
- **Documentation**: Automated doc building

### 4. Workflow Triggers

The pipeline runs on:
- **Push** to `main` or `develop` branches
- **Pull Requests** to `main`
- **Manual Triggers** via GitHub UI
- **Scheduled**: Daily dependency scans (2 AM UTC)

### 5. Branch Protection Setup

Configure branch protection rules for `main`:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

Required status checks:
- `Code Quality & Security`
- `Test Suite`
- `Build Package`

### 6. Integration with Autonomous System

The CI/CD pipeline integrates with the autonomous enhancement system:

```yaml
# Add to ci.yml for autonomous integration
- name: Run Autonomous Enhancement
  if: github.event_name == 'schedule'
  run: ./scripts/schedule_autonomous.sh run
```

## 🔧 Advanced Configuration

### Custom Workflow Modifications

You can customize the workflow by modifying these sections:

#### Python Version Matrix
```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]
```

#### Test Commands
```yaml
- name: Run tests
  run: |
    pytest tests/ -v --cov=src/autoformalize --cov-report=xml
```

#### Security Scanning
```yaml
- name: Security scan
  run: |
    bandit -r src/ -f json -o bandit-report.json
    safety check --json --output safety-report.json
```

### Environment Variables

Configure these environment variables in the workflow:

```yaml
env:
  PYTHON_DEFAULT_VERSION: "3.11"
  CACHE_VERSION: v1
```

## 📊 Monitoring & Maintenance

### Workflow Status

Monitor your CI/CD pipeline:

1. **Actions Tab**: View all workflow runs
2. **Status Badges**: Add to README.md
3. **Notifications**: Configure email/Slack alerts

### Performance Optimization

- **Cache Dependencies**: Pip cache reduces build time
- **Parallel Jobs**: Multiple jobs run concurrently  
- **Conditional Execution**: Skip unnecessary steps
- **Artifact Management**: Store and reuse build outputs

### Troubleshooting

Common issues and solutions:

#### Workflow Permission Errors
- Ensure repository has Actions enabled
- Check branch protection settings
- Verify secret configuration

#### Test Failures
- Review test logs in Actions tab
- Check for environment-specific issues
- Validate dependency compatibility

#### Security Scan Failures
- Review Bandit and Safety reports
- Update vulnerable dependencies
- Add security exceptions if needed

## 🎯 Next Steps

After setting up the workflow:

1. ✅ Monitor first few runs for issues
2. ✅ Configure branch protection rules
3. ✅ Set up status check requirements
4. ✅ Add repository secrets
5. ✅ Test release automation
6. ✅ Enable autonomous enhancement integration

The comprehensive CI/CD pipeline will provide:
- **Automated Quality Assurance**
- **Multi-platform Compatibility Testing**
- **Security Vulnerability Detection**
- **Automated Release Management**
- **Performance Regression Detection**

---

*Generated by Terragon Autonomous SDLC Enhancement System*  
*For questions or issues, see AUTONOMOUS_SDLC.md documentation*