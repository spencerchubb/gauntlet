#!/bin/bash

# If using Squarespace, make sure to remove their Squarespace Defaults in DNS settings.
# If Squarespace shows an error about saving DNS records, logging out and logging in could fix it.
# Make sure to add A record pointing to the server's IP address.

sudo snap install --classic certbot
sudo certbot -d gauntlet.spencerchubb.com --manual --preferred-challenges=dns certonly --key-type ecdsa