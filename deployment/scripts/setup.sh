#!/bin/bash
set -e

# Log file setup
LOGFILE="/var/log/freefontfinder_setup.log"
mkdir -p "$(dirname $LOGFILE)"
exec 1> >(tee -a "$LOGFILE") 2>&1

echo "Starting Free Font Finder setup at $(date)"

# Function to log steps
log_step() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Validation functions
validate_nginx() {
    if ! nginx -t; then
        echo "ERROR: nginx configuration test failed"
        return 1
    fi
    return 0
}

validate_services() {
    local services=("nginx" "freefontfinder")
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            echo "ERROR: Service $service is not running"
            return 1
        fi
    done
    return 0
}

validate_gunicorn() {
    local response=$(curl -s -w "%{http_code}" --max-time 10 http://localhost:8000/)
    local status=$?
    local http_code=${response: -3}
    
    if [ $status -ne 0 ]; then
        echo "ERROR: Curl failed with status $status"
        return 1
    fi
    
    if [ "$http_code" != "200" ]; then
        echo "ERROR: Gunicorn returned HTTP code $http_code instead of 200"
        return 1
    fi
    
    return 0
}


# 1. System Setup
log_step "Setting up swap space..."
if [ ! -f /swapfile ]; then
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
fi

# 2. Install Dependencies
log_step "Installing system dependencies..."
apt-get update
apt-get install -y python3-pip python3-venv nginx certbot python3-certbot-nginx git

# 3. Directory Structure Setup
log_step "Setting up directory structure..."
mkdir -p /var/www/freefontfinder/{static,model,logs,templates}
cd /var/www/freefontfinder

# 4. Python Environment
log_step "Setting up Python environment..."
# if [ -d "venv" ]; then
#     rm -rf venv
# fi
# python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r frontend_requirements.txt

# 5. Nginx Configuration
log_step "Setting up nginx configuration..."
ln -sf /etc/nginx/sites-available/freefontfinder /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

ln -sf /var/www/freefontfinder/deployment/configs/freefontfinder.service \
 /etc/systemd/system/freefontfinder.service

# Validate nginx configuration
if ! validate_nginx; then
    echo "ERROR: nginx configuration validation failed"
    exit 1
fi


# 8. Set Permissions
log_step "Setting permissions..."
chown -R www-data:www-data /var/www/freefontfinder
chown -R www-data:www-data /var/log/freefontfinder
chmod -R 755 /var/www/freefontfinder

# 9. SSL Setup
log_step "Setting up SSL with certbot..."
if host freefontfinder.com > /dev/null 2>&1; then
    log_step "DNS check passed, proceeding with SSL setup..."
    certbot --nginx \
        -d freefontfinder.com \
        -d www.freefontfinder.com \
        --non-interactive \
        --agree-tos \
        --redirect \
        --email holdsworthtimmy@yahoo.com
        
    # Verify SSL setup
    if curl -s -I https://freefontfinder.com | grep -q "200 OK"; then
        log_step "SSL setup successful!"
    else
        log_step "Warning: SSL setup might not be complete. Please check configuration."
    fi
else
    log_step "DNS not yet configured for freefontfinder.com - skipping SSL setup"
    log_step "Run: certbot --nginx -d freefontfinder.com -d www.freefontfinder.com once DNS is ready"
fi

# Setup auto-renewal
systemctl enable certbot.timer
systemctl start certbot.timer

# 10. Start Services
log_step "Starting services..."
systemctl daemon-reload
systemctl enable freefontfinder
systemctl restart freefontfinder
systemctl restart nginx

# 11. Validate Services
log_step "Validating services..."
sleep 5  # Give services time to start
if ! validate_services; then
    echo "ERROR: Service validation failed"
    # Show logs for debugging
    echo "nginx error log:"
    tail -n 20 /var/log/nginx/error.log
    echo "freefontfinder error log:"
    tail -n 20 /var/log/freefontfinder/error.log
    exit 1
fi

# 12. Validate Gunicorn
log_step "Validating Gunicorn..."
if ! validate_gunicorn; then
    echo "ERROR: Gunicorn validation failed"
    exit 1
fi

# 13. Memory Monitoring
log_step "Setting up memory monitoring..."
cat > /usr/local/bin/monitor_memory.sh << 'MONITOR_EOF'
#!/bin/bash
MEMORY_USAGE=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "High memory usage alert: ${MEMORY_USAGE}%" | logger -t memory_monitor
fi
MONITOR_EOF

chmod +x /usr/local/bin/monitor_memory.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/monitor_memory.sh") | crontab -

log_step "Setup completed successfully!"
echo "Memory Usage After Setup:"
free -h
echo
echo "Service Status:"
systemctl status nginx --no-pager
systemctl status freefontfinder --no-pager
echo
echo "Nginx Configuration Test:"
nginx -t
echo
echo "You can now access your site at: https://freefontfinder.com"