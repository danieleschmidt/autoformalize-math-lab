# Runbooks

This directory contains operational runbooks for the autoformalize-math-lab project. Runbooks provide step-by-step procedures for handling common operational scenarios, incidents, and maintenance tasks.

## Structure

- `incident-response.md` - Incident response procedures and escalation paths
- `low-success-rate.md` - Handling formalization success rate degradation
- `performance-issues.md` - Diagnosing and resolving performance problems
- `llm-api-issues.md` - Handling LLM API connectivity and quota issues
- `proof-assistant-issues.md` - Troubleshooting proof assistant problems
- `deployment-procedures.md` - Standard deployment and rollback procedures
- `maintenance-tasks.md` - Regular maintenance and housekeeping procedures
- `backup-recovery.md` - Data backup and recovery procedures

## Quick Reference

### Emergency Contacts

- **Platform Team**: #platform-team (Slack)
- **ML Engineering**: #ml-engineering (Slack)  
- **On-call Engineer**: Use PagerDuty escalation
- **Business Owner**: #product-team (Slack)

### Common Commands

```bash
# Check service status
kubectl get pods -l app=autoformalize

# View recent logs
kubectl logs -l app=autoformalize --tail=100

# Check metrics
curl http://prometheus:9090/api/v1/query?query=formalization_success_rate

# Emergency restart
kubectl rollout restart deployment/autoformalize
```

## Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| Critical | 15 minutes | Platform Team → Engineering Manager → VP Engineering |
| High | 1 hour | Platform Team → Engineering Manager |
| Medium | 4 hours | Platform Team |
| Low | 24 hours | Platform Team |

## Best Practices

1. **Document Everything**: Always document actions taken during incidents
2. **Communicate Early**: Update stakeholders proactively
3. **Follow Procedures**: Stick to established runbooks unless exceptional circumstances
4. **Post-Incident Reviews**: Conduct thorough post-mortems for all incidents
5. **Update Runbooks**: Keep procedures current based on learnings