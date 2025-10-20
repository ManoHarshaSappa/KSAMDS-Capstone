output "db_endpoint" {
  value       = aws_db_instance.ksamds.endpoint
  description = "The connection endpoint for the RDS instance"
}