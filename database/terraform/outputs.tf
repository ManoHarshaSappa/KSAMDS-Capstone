output "db_endpoint" {
  value       = aws_db_instance.ksamds.endpoint
  description = "The connection endpoint for the RDS instance"
}

output "db_port" {
  value       = aws_db_instance.ksamds.port
  description = "The port the RDS instance is listening on"
}

output "db_name" {
  value       = aws_db_instance.ksamds.db_name
  description = "The name of the default database"
}