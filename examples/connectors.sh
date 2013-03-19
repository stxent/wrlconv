src="connectors"
pattern=( "PLS2.*" "PLS3.*" "PLS4.*" "PLS5.*" "PLS6.*" "PLS8.*" "PLS10.*" "PLD6.*" "PLD8.*" "PLD10.*" "PLD12.*" "PLD14.*" "PLD16.*" "PLD18.*" "PLD20.*" "BH10.*" "BH14.*" "BH16.*" "BH20.*" "MJ-2.*" "MJ-3.*" )
#Coordinates of model center
offsetx=(      0.0      3.5      8.0     17.5     24.0     32.0     42.0     -11.5     -5.0    -10.5      -10.0      -2.5      -9.0       0.5      -8.0    -18.0    -19.0     -19.5   -20.5     13.0     13.5 )
offsety=(      0.0      0.0      0.0      0.0      0.0      0.0      0.0       0.0      0.0      0.0        0.0       0.0       0.0       0.0       0.0      0.0      0.0       0.0     0.0      0.0      0.0 )
offsetz=(      0.0      0.0      0.0      0.0      0.0      0.0      0.0       6.0      0.0      0.0       -6.0      -6.0     -12.0     -12.0     -18.0      0.0     -6.0     -12.0   -18.0      0.0     -6.0 )
dest=(     "pls-2"  "pls-3"  "pls-4"  "pls-5"  "pls-6"  "pls-8"  "pls-10"  "pld-6"  "pld-8"  "pld-10"  "pld-12"  "pld-14"  "pld-16"  "pld-18"  "pld-20"  "bh-10"  "bh-14"  "bh-16"  "bh-20"   "mj-2"   "mj-3" )
mkdir rebuild
for (( i = 0 ; i < ${#dest[@]} ; i++ ))
do
  #Coordinates rotated
  x=`echo "-1 * ${offsetx[$i]}" | bc -q`
  y=`echo "-1 * ${offsety[$i]}" | bc -q`
  z=`echo "-1 * ${offsetz[$i]}" | bc -q`
  wrload.py -w $src.x3d -f ${pattern[$i]} -t=$x,$y,$z
  mv $src.re.wrl rebuild/${dest[$i]}.wrl
  echo "${dest[$i]} done"
done
