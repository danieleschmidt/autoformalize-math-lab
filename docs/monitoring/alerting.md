# Alerting Configuration

## Overview

This document outlines alerting strategies and configurations for monitoring the autoformalize-math-lab application's health, performance, and operational metrics.

## Alert Categories

### Business Metrics Alerts

#### Formalization Success Rate
```yaml
# Low success rate alert
- alert: LowFormalizationSuccessRate
  expr: formalization_success_rate{target_system="lean4"} < 0.70
  for: 10m
  labels:
    severity: warning
    team: ml-engineering
    runbook: "https://docs.company.com/runbooks/low-success-rate"
  annotations:
    summary: "Formalization success rate below threshold"
    description: |
      Lean4 formalization success rate has been below 70% for more than 10 minutes.
      Current rate: {{ $value | humanizePercentage }}
      Target system: {{ $labels.target_system }}
      Domain: {{ $labels.domain }}

- alert: CriticalFormalizationSuccessRate
  expr: formalization_success_rate < 0.50
  for: 5m
  labels:
    severity: critical
    team: ml-engineering
    escalation: "pager"
  annotations:
    summary: "Critical formalization success rate"
    description: |
      Formalization success rate has dropped below 50% for {{ $labels.target_system }}.
      This requires immediate attention.
```

#### High Error Rates
```yaml
- alert: HighFormalizationErrorRate
  expr: |
    rate(formalizations_total{status="error"}[5m]) > 0.1
  for: 3m
  labels:
    severity: warning
    team: platform
  annotations:
    summary: "High formalization error rate detected"
    description: |
      Error rate is {{ $value | humanize }} errors per second
      Target system: {{ $labels.target_system }}
      Domain: {{ $labels.domain }}

- alert: FormalizationErrorSpike
  expr: |
    rate(formalizations_total{status="error"}[5m]) > 
    4 * rate(formalizations_total{status="error"}[1h] offset 1h)
  for: 2m
  labels:
    severity: critical
    team: platform
  annotations:
    summary: "Sudden spike in formalization errors"
    description: "Error rate has increased 4x compared to the same time yesterday"
```

### Performance Alerts

#### Response Time Degradation
```yaml
- alert: SlowFormalizationPerformance
  expr: |
    histogram_quantile(0.95, 
      rate(formalization_duration_seconds_bucket[10m])
    ) > 120
  for: 15m
  labels:
    severity: warning
    team: performance
  annotations:
    summary: "Slow formalization performance"
    description: |
      95th percentile formalization time is {{ $value }}s
      Target system: {{ $labels.target_system }}
      Stage: {{ $labels.stage }}

- alert: ExtremelySlowFormalization
  expr: |
    histogram_quantile(0.95, 
      rate(formalization_duration_seconds_bucket[5m])
    ) > 300
  for: 5m
  labels:
    severity: critical
    team: performance
  annotations:
    summary: "Extremely slow formalization performance"
    description: "95th percentile formalization time exceeds 5 minutes"
```

#### Proof Verification Timeouts
```yaml
- alert: HighVerificationTimeouts
  expr: |
    rate(proof_verification_attempts_total{result="timeout"}[10m]) > 0.05
  for: 5m
  labels:
    severity: warning
    team: infrastructure
  annotations:
    summary: "High proof verification timeout rate"
    description: |
      Verification timeout rate is {{ $value | humanize }} per second
      System: {{ $labels.system }}
```

### Resource and Infrastructure Alerts

#### System Resource Usage
```yaml
- alert: HighActiveFormalizationCount
  expr: active_formalizations > 100
  for: 10m
  labels:
    severity: warning
    team: infrastructure
  annotations:
    summary: "High number of concurrent formalizations"
    description: "{{ $value }} active formalizations may indicate resource constraints"

- alert: FormalizationQueueBacklog
  expr: formalization_queue_size > 1000
  for: 15m
  labels:
    severity: warning
    team: platform
  annotations:
    summary: "Large formalization queue backlog"
    description: "Queue has {{ $value }} pending formalizations"
```

#### Proof Assistant Availability
```yaml
- alert: ProofAssistantDown
  expr: up{job="proof-assistant"} == 0
  for: 1m
  labels:
    severity: critical
    team: infrastructure
    escalation: "pager"
  annotations:
    summary: "Proof assistant service is down"
    description: |
      Proof assistant {{ $labels.instance }} has been down for more than 1 minute.
      System: {{ $labels.system }}

- alert: ProofAssistantHighLatency
  expr: |
    histogram_quantile(0.95, 
      rate(proof_verification_duration_seconds_bucket[5m])
    ) > 60
  for: 10m
  labels:
    severity: warning
    team: infrastructure
  annotations:
    summary: "High proof verification latency"
    description: "95th percentile verification time is {{ $value }}s"
```

### Cost and Resource Management

#### LLM Cost Alerts
```yaml
- alert: HighLLMCosts
  expr: increase(llm_cost_usd_total[1h]) > 100
  for: 0m
  labels:
    severity: warning
    team: ml-engineering
  annotations:
    summary: "High LLM usage costs"
    description: |
      LLM costs exceeded $100 in the last hour
      Provider: {{ $labels.provider }}
      Model: {{ $labels.model }}
      Total cost: ${{ $value }}

- alert: LLMCostSpike
  expr: |
    increase(llm_cost_usd_total[10m]) > 
    3 * (increase(llm_cost_usd_total[1h] offset 1h) / 6)
  for: 0m
  labels:
    severity: critical
    team: ml-engineering
  annotations:
    summary: "Sudden spike in LLM costs"
    description: "LLM costs are significantly higher than expected"

- alert: LLMRateLimit
  expr: increase(llm_requests_total{status="rate_limited"}[5m]) > 10
  for: 0m
  labels:
    severity: warning
    team: ml-engineering
  annotations:
    summary: "LLM API rate limiting detected"
    description: "{{ $value }} rate-limited requests in the last 5 minutes"
```

#### Token Usage Monitoring
```yaml
- alert: HighTokenUsage
  expr: |
    rate(llm_tokens_used_total[1h]) > 1000000
  for: 30m
  labels:
    severity: warning
    team: ml-engineering
  annotations:
    summary: "High token usage rate"
    description: |
      Token usage rate is {{ $value }} tokens per second
      Provider: {{ $labels.provider }}
      Model: {{ $labels.model }}
```

### Data Quality and Pipeline Health

#### Parsing Errors
```yaml
- alert: HighParsingErrorRate
  expr: |
    rate(parsing_errors_total[10m]) > 0.05
  for: 5m
  labels:
    severity: warning
    team: data-engineering
  annotations:
    summary: "High LaTeX parsing error rate"
    description: |
      Parsing error rate is {{ $value | humanize }} errors per second
      Error type: {{ $labels.error_type }}

- alert: UnknownParsingErrors
  expr: |
    rate(parsing_errors_total{error_type="unknown"}[15m]) > 0.01
  for: 10m
  labels:
    severity: warning
    team: data-engineering
  annotations:
    summary: "Unknown parsing errors detected"
    description: "Encountering parsing errors of unknown type"
```

#### Correction Round Alerts
```yaml
- alert: ExcessiveCorrectionRounds
  expr: |
    histogram_quantile(0.90, 
      rate(correction_rounds_bucket[15m])
    ) > 5
  for: 10m
  labels:
    severity: warning
    team: ml-engineering
  annotations:
    summary: "High number of correction rounds"
    description: |
      90th percentile correction rounds is {{ $value }}
      This may indicate prompt quality issues
```

## Alertmanager Configuration

```yaml
# alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'
  
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default-receiver'
  routes:
    - match:
        escalation: pager
      receiver: 'pager-receiver'
      group_wait: 10s
      group_interval: 1m
      repeat_interval: 1h
    - match:
        severity: critical
      receiver: 'slack-critical'
      group_wait: 30s
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'team@company.com'
        subject: 'Autoformalize Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

  - name: 'slack-critical'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          🚨 {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}

  - name: 'slack-warnings'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: 'Warning: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          ⚠️ {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}

  - name: 'pager-receiver'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .Annotations.summary }}'
```

## Runbook Templates

### Low Success Rate Runbook

```markdown
# Runbook: Low Formalization Success Rate

## Alert Description
Formalization success rate has dropped below acceptable thresholds.

## Immediate Actions
1. Check recent deployments or configuration changes
2. Verify proof assistant availability and health
3. Review recent error logs for patterns
4. Check LLM API status and quotas

## Investigation Steps
1. Query error logs: `kubectl logs -l app=autoformalize | grep ERROR`
2. Check metrics dashboard for anomalies
3. Verify external dependencies (LLM APIs, proof assistants)
4. Review recent input data quality

## Resolution Steps
1. If LLM API issues: Switch to backup provider or model
2. If proof assistant issues: Restart services or scale up
3. If data quality issues: Enable additional input validation
4. If infrastructure issues: Scale resources or failover

## Prevention
- Implement better input validation
- Add more comprehensive monitoring
- Set up automated fallback mechanisms
```

## Alert Testing

```bash
#!/bin/bash
# scripts/test_alerts.sh

# Test alert by triggering high error rate
echo "Testing high error rate alert..."
for i in {1..50}; do
  curl -X POST http://localhost:8080/formalize \
    -H "Content-Type: application/json" \
    -d '{"latex": "invalid latex to trigger error"}' \
    -w "%{http_code}\n" -o /dev/null
done

# Test performance alert by simulating slow operations
echo "Testing performance alert..."
curl -X POST http://localhost:8080/formalize \
  -H "Content-Type: application/json" \
  -d '{"latex": "extremely complex proof", "slow_mode": true}'

echo "Check Prometheus alerts: http://localhost:9090/alerts"
echo "Check Alertmanager: http://localhost:9093"
```

## Best Practices

1. **Alert Fatigue**: Tune thresholds to minimize false positives
2. **Severity Levels**: Use appropriate severity for different alert types
3. **Runbooks**: Include runbook links in alert annotations
4. **Testing**: Regularly test alert configurations
5. **Documentation**: Document alert rationale and expected actions
6. **Escalation**: Set up proper escalation paths for critical alerts
7. **Correlation**: Group related alerts to reduce noise
8. **Time Windows**: Use appropriate time windows for different metrics