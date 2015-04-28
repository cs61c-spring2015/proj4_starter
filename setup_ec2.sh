if [ $(hostname -d) == "ec2.internal" ]
then
  # install cython
  sudo easy_install cython
else
  echo "not on ec2!"
fi
