variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "db_username" {
  description = "Database administrator username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database administrator password"
  type        = string
  sensitive   = true
}

variable "aws_session_token" {
  description = "AWS session token for Learner Lab"
  type        = string
  sensitive   = true
}

data "aws_availability_zones" "available" {
  state = "available"
}