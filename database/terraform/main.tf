terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region     = var.aws_region
  access_key = var.aws_access_key_id
  secret_key = var.aws_secret_access_key
  token      = var.aws_session_token
}

# VPC for RDS
resource "aws_vpc" "ksamds_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "ksamds-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ksamds_igw" {
  vpc_id = aws_vpc.ksamds_vpc.id

  tags = {
    Name = "ksamds-igw"
  }
}

# Route table for public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.ksamds_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ksamds_igw.id
  }

  tags = {
    Name = "ksamds-public-rt"
  }
}

# Public subnet
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.ksamds_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  # Enable auto-assign public IP
  map_public_ip_on_launch = true

  tags = {
    Name = "ksamds-public-subnet-${count.index + 1}"
  }
}

# Associate public subnets with public route table
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Security group for RDS
resource "aws_security_group" "rds_sg" {
  name        = "ksamds-rds-sg"
  description = "Security group for KSAMDS RDS instance"
  vpc_id      = aws_vpc.ksamds_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # In production, restrict this to specific IPs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# RDS subnet group
resource "aws_db_subnet_group" "ksamds" {
  name       = "ksamds-subnet-group"
  subnet_ids = aws_subnet.public[*].id

  tags = {
    Name = "KSAMDS DB subnet group"
  }
}

# RDS instance
resource "aws_db_instance" "ksamds" {
  identifier           = "ksamds-db"
  engine              = "postgres"
  engine_version      = "15.3"
  instance_class      = "db.t3.micro"  # Free tier eligible
  allocated_storage   = 20
  storage_type        = "gp2"
  
  db_name             = "ksamds"
  username            = var.db_username
  password            = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.ksamds.name
  
  skip_final_snapshot    = true  # For development; set to false in production
  publicly_accessible    = true  # For development; set to false in production
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  tags = {
    Environment = "development"
    Project     = "KSAMDS"
  }
}
