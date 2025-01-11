If my configuration gets deleted for some reason, I can use these notes to recreate it.

### S3 Bucket Policy
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::spencer-chubb-gauntlet/*",
            "Condition": {
                "StringEquals": {
                    "s3:x-amz-acl": "public-read"
                }
            }
        }
    ]
}
```

### S3 CORS
```
[
    {
        "AllowedHeaders": [
            "Authorization",
            "Content-Type",
            "Content-Length",
            "*"
        ],
        "AllowedMethods": [
            "GET",
            "PUT",
            "POST",
            "DELETE",
            "HEAD"
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
