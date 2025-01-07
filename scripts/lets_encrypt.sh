#!/bin/bash

# If using Squarespace, make sure to remove their Squarespace Defaults in DNS settings.
# If Squarespace shows an error about saving DNS records, logging out and logging in could fix it.
# Make sure to add A record pointing to the server's IP address.

sudo snap install --classic certbot

DOMAIN1=pandapal.app
WILDCARD1=*.$DOMAIN1
DOMAIN2=pandapal.net
WILDCARD2=*.$DOMAIN2

echo $DOMAIN1 && echo $WILDCARD1 && echo $DOMAIN2 && echo $WILDCARD2

sudo certbot -d $DOMAIN1 -d $WILDCARD1 -d $DOMAIN2 -d $DOMAIN2 --manual --preferred-challenges=dns certonly --key-type ecdsa