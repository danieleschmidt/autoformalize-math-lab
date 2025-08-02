# Incident Response Procedures

## Overview

This runbook provides standardized procedures for responding to incidents in the autoformalize-math-lab system.

## Incident Classification

### Severity Levels

**Critical (SEV1)**
- Complete service outage
- Data loss or corruption
- Security breach
- Success rate below 10%

**High (SEV2)**
- Partial service degradation
- Success rate below 50%
- Performance degradation >300% of baseline
- API availability <95%

**Medium (SEV3)**
- Minor performance issues
- Success rate 50-70%
- Non-critical feature failures
- Monitoring alerts

**Low (SEV4)**
- Cosmetic issues
- Documentation problems
- Minor configuration issues

## Immediate Response (First 15 Minutes)

### 1. Assessment and Declaration

```bash
# Quick health check
curl -f http://autoformalize.example.com/health || echo "HEALTH CHECK FAILED"

# Check Prometheus metrics
curl -s "http://prometheus:9090/api/v1/query?query=up{job='autoformalize'}" | jq '.data.result[0].value[1]'

# Check recent deployments
kubectl rollout history deployment/autoformalize

# Check error rates
kubectl logs -l app=autoformalize --since=15m | grep -i error | wc -l
```

### 2. Initial Communication

```bash
# Create incident channel (use your incident management tool)
# Example Slack command:
/incident create "Autoformalize service degradation - investigating formalization failures"

# Update status page
curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
  -H "Authorization: OAuth TOKEN" \
  -d "incident[name]=Service Degradation" \
  -d "incident[status]=investigating"
```

### 3. Gather Initial Data

```bash
# Service status
kubectl get pods -l app=autoformalize -o wide

# Resource usage
kubectl top pods -l app=autoformalize

# Recent events
kubectl get events --sort-by='.lastTimestamp' | tail -20

# Check external dependencies
dig api.openai.com
curl -I https://api.anthropic.com/v1/health
```

## Detailed Investigation

### Performance Issues

```bash
# Check CPU and memory usage
kubectl top pods -l app=autoformalize

# Check database connections
kubectl exec -it deployment/autoformalize -- python -c "
from autoformalize.database import check_connection
print(check_connection())
"

# Check queue sizes
kubectl exec -it deployment/autoformalize -- python -c "
from autoformalize.queue import get_queue_size
print('Queue size:', get_queue_size())
"

# Review slow queries
kubectl logs -l app=autoformalize | grep "slow_query" | tail -10
```

### Success Rate Degradation

```bash
# Check recent formalization attempts
kubectl logs -l app=autoformalize --since=1h | grep "formalization_result" | tail -20

# Check LLM API status
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Check proof assistant availability
kubectl exec -it deployment/autoformalize -- lean --version
kubectl exec -it deployment/autoformalize -- isabelle version

# Review error patterns
kubectl logs -l app=autoformalize --since=1h | grep "ERROR" | cut -d' ' -f5- | sort | uniq -c | sort -nr
```

### External Dependencies

```bash
# Check LLM API quotas and limits
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  "https://api.openai.com/v1/usage?date=$(date +%Y-%m-%d)"

# Test proof assistant functionality
kubectl exec -it deployment/autoformalize -- python -c "
from autoformalize.verifiers import LeanVerifier
verifier = LeanVerifier()
result = verifier.verify('theorem test : True := trivial')
print('Lean verification:', result.success)
"

# Check network connectivity
kubectl exec -it deployment/autoformalize -- nslookup api.openai.com
kubectl exec -it deployment/autoformalize -- wget -q --spider https://api.anthropic.com
```

## Common Mitigation Strategies

### Quick Fixes

```bash
# Restart pods with issues
kubectl delete pod -l app=autoformalize,status=Failed

# Scale up for load issues
kubectl scale deployment autoformalize --replicas=10

# Switch to backup LLM provider
kubectl set env deployment/autoformalize LLM_PROVIDER=anthropic

# Enable circuit breaker for external calls
kubectl set env deployment/autoformalize CIRCUIT_BREAKER_ENABLED=true
```

### Rollback Procedures

```bash
# Check rollout history
kubectl rollout history deployment/autoformalize

# Rollback to previous version
kubectl rollout undo deployment/autoformalize

# Rollback to specific revision
kubectl rollout undo deployment/autoformalize --to-revision=3

# Monitor rollback progress
kubectl rollout status deployment/autoformalize
```

### Traffic Management

```bash
# Enable maintenance mode
kubectl patch configmap autoformalize-config --patch '{"data":{"maintenance_mode":"true"}}'

# Drain specific nodes
kubectl drain NODE_NAME --ignore-daemonsets --delete-emptydir-data

# Route traffic to healthy instances only
kubectl label pods -l app=autoformalize healthy=true
kubectl patch service autoformalize --patch '{"spec":{"selector":{"healthy":"true"}}}'
```

## Communication Templates

### Initial Incident Notification

```
🚨 INCIDENT ALERT: Autoformalize Service Degradation

STATUS: Investigating
SEVERITY: SEV2
IMPACT: Formalization success rate below 50%
START TIME: 14:30 UTC
RESPONDER: @oncall-engineer

We are investigating reports of degraded formalization performance. 
Users may experience slower response times and increased failure rates.

Updates will be provided every 15 minutes.
```

### Status Updates

```
📊 INCIDENT UPDATE: Autoformalize Service Degradation

STATUS: Identified root cause
IMPACT: Performance degradation due to LLM API rate limiting
ACTION: Implementing traffic throttling and backup provider failover
ETA: 30 minutes

Current metrics:
- Success rate: 65% (improving from 45%)
- Response time: 85th percentile at 45s
- Active formalizations: 23
```

### Resolution Notification

```
✅ INCIDENT RESOLVED: Autoformalize Service Degradation

STATUS: Resolved
DURATION: 45 minutes
ROOT CAUSE: LLM API rate limiting during peak traffic

RESOLUTION: 
- Implemented intelligent traffic throttling
- Activated backup LLM provider
- Scaled infrastructure to handle increased load

METRICS:
- Success rate: 89% (back to normal)
- Response time: 85th percentile at 12s
- All systems operational

POST-INCIDENT REVIEW: Scheduled for tomorrow 10:00 AM
```

## Post-Incident Procedures

### Immediate Actions (Within 24 Hours)

1. **Document Timeline**: Record all actions taken and their timestamps
2. **Preserve Evidence**: Save logs, metrics, and configuration snapshots
3. **Update Stakeholders**: Send final incident summary
4. **Review Monitoring**: Ensure alerts fired appropriately

### Post-Mortem Process (Within 1 Week)

1. **Schedule Review**: Book post-mortem meeting with all responders
2. **Create Timeline**: Detailed chronology of events and responses
3. **Root Cause Analysis**: 5-whys or similar methodology
4. **Action Items**: Specific, assignable improvements
5. **Documentation**: Update runbooks based on learnings

### Post-Mortem Template

```markdown
# Post-Incident Review: [Date] - [Brief Description]

## Summary
- **Date**: 
- **Duration**: 
- **Severity**: 
- **Root Cause**: 
- **Resolution**: 

## Timeline
- 14:30 - Initial reports of degraded performance
- 14:35 - Incident declared, investigation started
- 14:45 - Root cause identified (LLM API rate limiting)
- 15:00 - Mitigation deployed
- 15:15 - Service fully restored

## What Went Well
- Quick detection and alerting
- Effective communication to stakeholders
- Successful mitigation strategy

## What Could Be Improved
- Earlier detection of API rate limiting
- Automated failover to backup providers
- Better load testing of peak scenarios

## Action Items
1. [ ] Implement proactive API quota monitoring (@engineer, 2 weeks)
2. [ ] Add automated failover logic (@engineer, 1 week)
3. [ ] Enhance load testing scenarios (@qa, 3 weeks)
4. [ ] Update runbook with new procedures (@ops, 1 week)
```

## Emergency Contacts

### Primary Escalation
- **Platform Team Lead**: @platform-lead
- **Engineering Manager**: @eng-manager
- **On-Call Engineer**: Use PagerDuty

### Secondary Escalation
- **VP Engineering**: @vp-eng (for SEV1 only)
- **Product Manager**: @product-manager
- **Customer Success**: @customer-success

### External Vendors
- **Cloud Provider**: [Support Portal Link]
- **LLM Provider**: [Support Contact]
- **Monitoring Provider**: [Support Contact]

## Tools and Resources

- **Monitoring**: Grafana dashboard at http://grafana.internal/autoformalize
- **Logs**: Kibana at http://kibana.internal
- **Metrics**: Prometheus at http://prometheus.internal
- **Alerting**: AlertManager at http://alertmanager.internal
- **Status Page**: https://status.company.com
- **Incident Management**: [Your incident management tool]