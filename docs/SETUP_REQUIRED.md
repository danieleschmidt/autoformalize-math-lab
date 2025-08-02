# Manual Setup Required

This document outlines manual setup steps that need to be completed by repository maintainers due to GitHub App permission limitations.

## 🔧 Required Manual Actions

### 1. GitHub Actions Workflows

**Status**: ⚠️ Manual Setup Required

The comprehensive CI/CD workflows have been prepared but need manual activation:

```bash
# Copy the CI workflow template
cp docs/workflows/ci-template.yml .github/workflows/ci.yml

# Copy additional workflow templates as needed
cp docs/workflows/examples/security-template.yml .github/workflows/security.yml
cp docs/workflows/examples/release-template.yml .github/workflows/release.yml
cp docs/workflows/examples/performance-template.yml .github/workflows/performance.yml

# Commit the workflows
git add .github/workflows/
git commit -m "ci: activate comprehensive CI/CD pipeline"
git push
```

### 2. Repository Secrets Configuration

**Status**: ⚠️ Manual Setup Required

Configure the following secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

#### Required Secrets:
- `PYPI_API_TOKEN`: For automated package publishing to PyPI
- `CODECOV_TOKEN`: For coverage reporting integration
- `DOCKER_USERNAME`: For Docker Hub publishing
- `DOCKER_PASSWORD`: For Docker Hub authentication

#### Optional Secrets for Enhanced Features:
- `OPENAI_TEST_API_KEY`: For testing LLM integrations (use test account)
- `ANTHROPIC_TEST_API_KEY`: For testing Claude API features (use test account)
- `SLACK_WEBHOOK_URL`: For Slack notifications
- `SNYK_TOKEN`: For advanced security scanning

### 3. Branch Protection Rules

**Status**: ⚠️ Manual Setup Required

Configure branch protection for the `main` branch:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable the following settings:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (minimum 1)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from CODEOWNERS
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

#### Required Status Checks:
- `Code Quality`
- `Test Suite` 
- `Build Package`
- `Security Scan`

### 4. Repository Settings

**Status**: ⚠️ Manual Setup Required

Configure the following repository settings:

#### General Settings:
- **Description**: "LLM-driven auto-formalization workbench that converts LaTeX proofs into Lean4/Isabelle"
- **Website**: Your project homepage (if any)
- **Topics**: `mathematics`, `formal-verification`, `llm`, `lean4`, `isabelle`, `proof-assistant`

#### Features:
- ✅ Enable Issues
- ✅ Enable Projects  
- ✅ Enable Wiki
- ✅ Enable Discussions

#### Pull Requests:
- ✅ Allow merge commits
- ✅ Allow squash merging
- ✅ Allow rebase merging
- ✅ Always suggest updating pull request branches
- ✅ Automatically delete head branches

### 5. Issue and PR Templates

**Status**: ✅ Already Configured

Issue and PR templates are included in the `.github/` directory structure and will be automatically available.

### 6. GitHub Pages (Optional)

**Status**: ⚠️ Manual Setup Required

If you want to enable documentation hosting:

1. Go to Settings → Pages
2. Select "Deploy from a branch"
3. Choose "gh-pages" branch (will be created by docs workflow)
4. Select "/ (root)" folder
5. Click Save

### 7. Security Settings

**Status**: ⚠️ Manual Setup Required

Configure security features:

#### Vulnerability Alerts:
1. Go to Settings → Security & analysis
2. Enable "Dependency graph"
3. Enable "Dependabot alerts"
4. Enable "Dependabot security updates"

#### Secret Scanning:
1. Enable "Secret scanning" (if available)
2. Enable "Push protection" (if available)

### 8. Integration Setup

**Status**: ⚠️ Manual Setup Required

#### Codecov Integration:
1. Visit https://codecov.io/
2. Sign up/sign in with GitHub
3. Add your repository
4. Copy the repository token
5. Add as `CODECOV_TOKEN` secret

#### Docker Hub Integration (Optional):
1. Create Docker Hub account
2. Create repository: `your-username/autoformalize-math-lab`
3. Generate access token
4. Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets

### 9. Monitoring Setup (Optional)

**Status**: ⚠️ Manual Setup Required

For production monitoring setup:

#### Prometheus/Grafana:
1. Deploy Prometheus and Grafana
2. Import dashboard from `docs/monitoring/dashboards/grafana-dashboard.json`
3. Configure alert rules from `docs/monitoring/alerting.md`

#### Status Page:
1. Set up status page service
2. Configure incident management integration
3. Update monitoring documentation with URLs

## 📋 Verification Checklist

After completing manual setup, verify the following:

- [ ] All GitHub Actions workflows are active and passing
- [ ] Branch protection rules are enforced
- [ ] Required secrets are configured
- [ ] Issue and PR templates are available
- [ ] CODEOWNERS file is working (automatic review requests)
- [ ] Security scanning is active
- [ ] Documentation builds successfully
- [ ] Tests run in CI environment

## 🚨 Security Considerations

### Secrets Management:
- Never commit secrets to the repository
- Use environment-specific secrets (test vs production)
- Rotate API keys regularly
- Limit secret access to necessary workflows only

### Access Control:
- Regularly review repository access
- Use principle of least privilege
- Enable two-factor authentication for all contributors
- Monitor repository security advisories

## 📞 Support

If you encounter issues during setup:

1. Check the troubleshooting documentation in `docs/workflows/best-practices.md`
2. Review GitHub Actions logs for specific error messages
3. Verify all required secrets are correctly configured
4. Ensure branch protection rules are not blocking necessary operations

## 🔄 Maintenance

Regular maintenance tasks:

- **Weekly**: Review failed workflows and security alerts
- **Monthly**: Update workflow dependencies and secrets rotation
- **Quarterly**: Review and optimize repository settings
- **Annually**: Audit security configurations and access controls

---

*This document is automatically generated as part of the Terragon SDLC implementation. Update as needed based on your specific requirements.*