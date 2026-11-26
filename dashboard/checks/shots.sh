#!/bin/bash
# Screenshots of the dashboard with headless Chrome. Serves the folder on a free local port and stops the server after.
#   bash checks/shots.sh
set -e
cd "$(dirname "$0")/.."
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PY=${PYTHON:-python3}
PORT=$($PY -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])")
$PY -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1
mkdir -p shots
desk() { "$CH" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=5000 --window-size="$2" \
  --screenshot="shots/$1.png" "http://127.0.0.1:$PORT/${4:-index.html?}$3" >/dev/null 2>&1; }
# headless Chrome will not lay out narrower than 500 px, so the phone view is an iframe of the right size, cropped after
phone() { "$CH" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=5000 --force-device-scale-factor=2 \
  --window-size="600,$2" --screenshot="shots/$1.png" "http://127.0.0.1:$PORT/checks/phone.html?w=390&h=$2&$3" >/dev/null 2>&1
  $PY -c "from PIL import Image;import sys;p=sys.argv[1];im=Image.open(p);im.crop((0,0,780,im.height)).save(p)" "shots/$1.png"; }
desk elec_1440x900 1440,900 "tab=elec"
desk opal_1440x900 1440,900 "tab=opal&range=all"
# a scrolled page comes out blank in a headless screenshot, so the scrolled views also go through the iframe
desk elec_tryit_1440x900 1440,900 "tab=elec&drop=10&days=20&focus=tryit" "checks/phone.html?w=1440&h=900&"
desk opal_tryit_1440x900 1440,900 "tab=opal&drop=20&days=20&focus=tryit" "checks/phone.html?w=1440&h=900&"
desk elec_full_1440 1440,3000 "tab=elec&range=365"
desk opal_full_1440 1440,3000 "tab=opal&range=all"
phone elec_390x844 844 "tab=elec"
phone opal_390x844 844 "tab=opal&range=365"
phone elec_tryit_390x844 844 "tab=elec&drop=10&days=20&focus=tryit"
phone opal_tryit_390x844 844 "tab=opal&drop=20&days=20&focus=tryit"
phone elec_full_390 4600 "tab=elec"
phone opal_full_390 5400 "tab=opal&range=all"
ls shots
