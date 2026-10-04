"""Owner's 0.1% tolerance amendment; keep the completed exact-match results intact."""
from fetch_matched_common import *
OLD_ROOT=ROOT;OLD_PACKET=PACKET
ROOT=OLD_ROOT/'tolerance_001';PACKET=OLD_PACKET/'tolerance_001'
ROOT.mkdir(exist_ok=True);PACKET.mkdir(exist_ok=True)
def eligible(br,ca):
 # Integer arithmetic implements the inclusive BR-relative 0.1% bound.
 return all(abs(br[k]-ca[k])*1000<=br[k] for k in ['total_fetches','H2D_bytes'])
