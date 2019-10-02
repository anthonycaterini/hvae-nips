# List of tests for the HVAE paper. 
# It is NOT recommended to run all of these at once.

# Note: If you want to receive an email at the end
# of the test, include the '--sender' and '--receiver'
# flags with the test.

# VAE
python3 main.py -m vae -s 85547
python3 main.py -m vae -s 12345
python3 main.py -m vae -s 98765

# HVAE, K = 1
python3 main.py -m hvae -K 1 -T fixed -s 85547 -ve false
python3 main.py -m hvae -K 1 -T fixed -s 12345 -ve false
python3 main.py -m hvae -K 1 -T none -s 85547 -ve false
python3 main.py -m hvae -K 1 -T none -s 12345 -ve false

# HVAE, K = 5
python3 main.py -m hvae -K 5 -T free -s 85547 -ve false
python3 main.py -m hvae -K 5 -T free -s 12345 -ve false
python3 main.py -m hvae -K 5 -T free -s 85547 -ve true
python3 main.py -m hvae -K 5 -T free -s 12345 -ve true
python3 main.py -m hvae -K 5 -T fixed -s 85547 -ve false
python3 main.py -m hvae -K 5 -T fixed -s 12345 -ve false
python3 main.py -m hvae -K 5 -T fixed -s 85547 -ve true
python3 main.py -m hvae -K 5 -T fixed -s 12345 -ve true
python3 main.py -m hvae -K 5 -T none -s 85547 -ve false
python3 main.py -m hvae -K 5 -T none -s 12345 -ve false
python3 main.py -m hvae -K 5 -T none -s 85547 -ve true
python3 main.py -m hvae -K 5 -T none -s 12345 -ve true

# HVAE, K = 10
python3 main.py -m hvae -K 10 -T free -s 85547 -ve false
python3 main.py -m hvae -K 10 -T free -s 12345 -ve false
python3 main.py -m hvae -K 10 -T free -s 98765 -ve false
python3 main.py -m hvae -K 10 -T free -s 85547 -ve true
python3 main.py -m hvae -K 10 -T free -s 12345 -ve true
python3 main.py -m hvae -K 10 -T free -s 98765 -ve true
python3 main.py -m hvae -K 10 -T fixed -s 85547 -ve false
python3 main.py -m hvae -K 10 -T fixed -s 12345 -ve false
python3 main.py -m hvae -K 10 -T fixed -s 85547 -ve true
python3 main.py -m hvae -K 10 -T fixed -s 12345 -ve true
python3 main.py -m hvae -K 10 -T none -s 85547 -ve false
python3 main.py -m hvae -K 10 -T none -s 12345 -ve false
python3 main.py -m hvae -K 10 -T none -s 85547 -ve true
python3 main.py -m hvae -K 10 -T none -s 12345 -ve true

# HVAE, K = 15
python3 main.py -m hvae -K 15 -T free -s 85547 -ve false
python3 main.py -m hvae -K 15 -T free -s 12345 -ve false
python3 main.py -m hvae -K 15 -T free -s 85547 -ve true
python3 main.py -m hvae -K 15 -T free -s 12345 -ve true
python3 main.py -m hvae -K 15 -T fixed -s 85547 -ve false
python3 main.py -m hvae -K 15 -T fixed -s 12345 -ve false
python3 main.py -m hvae -K 15 -T fixed -s 85547 -ve true
python3 main.py -m hvae -K 15 -T fixed -s 12345 -ve true
python3 main.py -m hvae -K 15 -T none -s 85547 -ve false
python3 main.py -m hvae -K 15 -T none -s 12345 -ve false
python3 main.py -m hvae -K 15 -T none -s 85547 -ve true
python3 main.py -m hvae -K 15 -T none -s 12345 -ve true

# HVAE, K = 20
python3 main.py -m hvae -K 20 -T free -s 85547 -ve false
python3 main.py -m hvae -K 20 -T free -s 12345 -ve false
python3 main.py -m hvae -K 20 -T free -s 85547 -ve true
python3 main.py -m hvae -K 20 -T free -s 12345 -ve true
python3 main.py -m hvae -K 20 -T fixed -s 85547 -ve false
python3 main.py -m hvae -K 20 -T fixed -s 12345 -ve false
python3 main.py -m hvae -K 20 -T fixed -s 85547 -ve true
python3 main.py -m hvae -K 20 -T fixed -s 12345 -ve true
python3 main.py -m hvae -K 20 -T none -s 85547 -ve false
python3 main.py -m hvae -K 20 -T none -s 12345 -ve false
python3 main.py -m hvae -K 20 -T none -s 85547 -ve true
python3 main.py -m hvae -K 20 -T none -s 12345 -ve true
