If my configuration gets deleted for some reason, I can use these notes to recreate it.

### S3 CORS
```
[
    {
        "AllowedHeaders": [
            "Authorization",
            "Content-Type",
            "Content-Length"
        ],
        "AllowedMethods": [
            "GET",
            "PUT"
        ],
        "AllowedOrigins": [
            "http://localhost:8000",
            "https://gauntlet.spencerchubb.com"
        ],
        "ExposeHeaders": [
            "ETag"
        ],
        "MaxAgeSeconds": 3000
    }
]
```

### EC2 Instance
- Name: spencer-chubb-ec2
- OS: Ubuntu 24.04
- Instance type: t3.micro
- Key pair: spencer-chubb-key
- Security group: launch-wizard-1
- Storage: default
