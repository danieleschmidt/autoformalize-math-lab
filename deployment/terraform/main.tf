# Terraform configuration for multi-region Autoformalize deployment
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket = "autoformalize-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-west-2"
    
    dynamodb_table = "autoformalize-terraform-locks"
    encrypt        = true
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.primary_region
  
  default_tags {
    tags = {
      Project     = "Autoformalize"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "Terragon Labs"
    }
  }
}

# Configure AWS Provider for secondary region
provider "aws" {
  alias  = "secondary"
  region = var.secondary_region
  
  default_tags {
    tags = {
      Project     = "Autoformalize"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = "Terragon Labs"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-west-2"
}

variable "secondary_region" {
  description = "Secondary AWS region for DR"
  type        = string
  default     = "us-east-1"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "autoformalize-prod"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 3
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 20
}

variable "desired_nodes" {
  description = "Desired number of nodes"
  type        = number
  default     = 6
}

# Data sources
data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = var.availability_zones
  private_subnets = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, i)]
  public_subnets  = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, i + 100)]
  
  enable_nat_gateway     = true
  single_nat_gateway     = false
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true
  
  # Enable flow logs for security monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
  
  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# Security Groups
resource "aws_security_group" "additional" {
  name_prefix = "${var.cluster_name}-additional"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  tags = {
    Name = "${var.cluster_name}-additional"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  control_plane_subnet_ids       = module.vpc.private_subnets
  
  # Cluster endpoint configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = ["0.0.0.0/0"]
  
  # Enable logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Cluster security group
  cluster_additional_security_group_ids = [aws_security_group.additional.id]
  
  # Node groups
  eks_managed_node_groups = {
    general = {
      desired_size = var.desired_nodes
      max_size     = var.max_nodes
      min_size     = var.min_nodes
      
      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        Application = "autoformalize"
      }
      
      update_config = {
        max_unavailable_percentage = 25
      }
      
      # Use custom launch template
      create_launch_template = true
      launch_template_name   = "${var.cluster_name}-node-group"
      
      pre_bootstrap_user_data = <<-EOT
        #!/bin/bash
        yum update -y
        yum install -y awscli
      EOT
      
      # Taints for dedicated nodes
      taints = [
        {
          key    = "autoformalize.io/dedicated"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    spot = {
      desired_size = 2
      max_size     = 10
      min_size     = 0
      
      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        Application = "autoformalize"
        NodeType    = "spot"
      }
      
      taints = [
        {
          key    = "spot"
          value  = "true"  
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # OIDC Identity provider
  cluster_identity_providers = {
    sts = {
      client_id = "sts.amazonaws.com"
    }
  }
  
  tags = {
    Environment = var.environment
  }
}

# RDS for persistent storage
resource "aws_db_subnet_group" "autoformalize" {
  name       = "${var.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Name = "${var.cluster_name} DB subnet group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_name}-rds"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-rds"
  }
}

resource "aws_db_instance" "autoformalize" {
  identifier = "${var.cluster_name}-postgres"
  
  engine         = "postgres"
  engine_version = "15"
  instance_class = "db.t3.medium"
  
  allocated_storage       = 100
  max_allocated_storage   = 1000
  storage_type           = "gp2"
  storage_encrypted      = true
  
  db_name  = "autoformalize"
  username = "autoformalize"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.autoformalize.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-postgres-final-snapshot"
  
  # Enable automatic minor version upgrades
  auto_minor_version_upgrade = true
  
  # Performance Insights
  performance_insights_enabled = true
  
  tags = {
    Name = "${var.cluster_name}-postgres"
  }
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# ElastiCache for Redis
resource "aws_elasticache_subnet_group" "autoformalize" {
  name       = "${var.cluster_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "elasticache" {
  name_prefix = "${var.cluster_name}-cache"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  tags = {
    Name = "${var.cluster_name}-cache"
  }
}

resource "aws_elasticache_replication_group" "autoformalize" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "Redis cluster for Autoformalize"
  
  port               = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 3
  node_type         = "cache.t3.micro"
  
  subnet_group_name  = aws_elasticache_subnet_group.autoformalize.name
  security_group_ids = [aws_security_group.elasticache.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  # Backup configuration
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  # Maintenance window
  maintenance_window = "sun:05:00-sun:07:00"
  
  tags = {
    Name = "${var.cluster_name}-redis"
  }
}

# S3 bucket for file storage
resource "aws_s3_bucket" "autoformalize_data" {
  bucket = "${var.cluster_name}-data-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.cluster_name}-data"
    Environment = var.environment
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "autoformalize_data" {
  bucket = aws_s3_bucket.autoformalize_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "autoformalize_data" {
  bucket = aws_s3_bucket.autoformalize_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "autoformalize_data" {
  bucket = aws_s3_bucket.autoformalize_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "autoformalize" {
  name              = "/aws/eks/${var.cluster_name}/application"
  retention_in_days = 30
  
  tags = {
    Environment = var.environment
    Application = "autoformalize"
  }
}

# IAM roles and policies
resource "aws_iam_role" "autoformalize_pod_role" {
  name = "${var.cluster_name}-pod-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub" = "system:serviceaccount:autoformalize:autoformalize-sa"
            "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "autoformalize_policy" {
  name        = "${var.cluster_name}-policy"
  description = "Policy for Autoformalize application"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.autoformalize_data.arn,
          "${aws_s3_bucket.autoformalize_data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:secretsmanager:${var.primary_region}:${data.aws_caller_identity.current.account_id}:secret:autoformalize/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "${aws_cloudwatch_log_group.autoformalize.arn}:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "autoformalize_policy_attachment" {
  role       = aws_iam_role.autoformalize_pod_role.name
  policy_arn = aws_iam_policy.autoformalize_policy.arn
}

# Application Load Balancer
resource "aws_lb" "autoformalize" {
  name               = "${var.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  # Access logs
  access_logs {
    bucket  = aws_s3_bucket.autoformalize_logs.bucket
    prefix  = "alb"
    enabled = true
  }
  
  tags = {
    Name = "${var.cluster_name}-alb"
  }
}

resource "aws_security_group" "alb" {
  name_prefix = "${var.cluster_name}-alb"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-alb"
  }
}

# S3 bucket for ALB logs
resource "aws_s3_bucket" "autoformalize_logs" {
  bucket = "${var.cluster_name}-logs-${random_id.logs_bucket_suffix.hex}"
  
  tags = {
    Name = "${var.cluster_name}-logs"
  }
}

resource "random_id" "logs_bucket_suffix" {
  byte_length = 4
}

# WAF for application protection
resource "aws_wafv2_web_acl" "autoformalize" {
  name  = "${var.cluster_name}-waf"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    override_action {
      none {}
    }
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
    
    action {
      block {}
    }
  }
  
  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.cluster_name}WAF"
    sampled_requests_enabled   = true
  }
  
  tags = {
    Name = "${var.cluster_name}-waf"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "kubectl_config" {
  description = "kubectl config as generated by the module"
  value = {
    cluster_name = module.eks.cluster_name
    endpoint     = module.eks.cluster_endpoint
    ca_data      = module.eks.cluster_certificate_authority_data
  }
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.autoformalize.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.autoformalize.primary_endpoint_address
}

output "s3_bucket" {
  description = "S3 bucket name for data storage"
  value       = aws_s3_bucket.autoformalize_data.bucket
}

output "load_balancer_dns" {  
  description = "DNS name of the load balancer"
  value       = aws_lb.autoformalize.dns_name
}