# rust-threshold-signature
   threshold-signature is a signature that can adjust the minimum number of the people to vote for a proposal easily, the protocal of the signature is the same as ecdsa, but the private key is the constant of a polynomial````f(x) = private key + a1x + a2x**2 ...````, so we can change the degree of polynomial to limit how many person can recover the private key, and although this person work together to make a signature that contain private key, but each of them doesn't know the acutal private key.

## paillier key share
  so, to do this, we need a function that allow two person to generate a key, but both of them doesn't know the key itself, is call multiplication to add. In this situation Allice has a key share A, Bob has B, Allice use paillier encryption (it's likes rsa, but it contain ZK verify) send  ````encryption(A)````, Bob recive the cipher then he mut B to the cipher and also add C to it, then he keep -C as key share and send ````cipher(A*B + C)````to Allice, so after Allice dencrypt the cipher, she got ````A*B + C````, so they can add the key share to get  ````A*B + C +(-C) = A*B````

## make public
  using paillier key share, they can make a function ````f(x) = private key + a1x + a2x**2 ...````, and after doing operating of ecc cruve, they will have ````f(x) = pubkey + a1x*G + (a2x**2)*G ...````, then they can calculate pubkey (although they can calculate private key also but they shouldn't)

