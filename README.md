# rust-threshold-signature
   what we need to do for changing the minimum threshold for a team to make a proposal or vote, threshold-signature can do that easily, you just need to change the public key of the group, and it is reusable if the threshold doesn't change. With this project we can detact who has been hack to send the wrong signature for the protocal, and even doesn't need a merkle tree to proof the member is in the group, next I will explame step by step from making a public key, do some secret exchange for the signature, how we verify the message for secret exchange, and detect who has been hack, and finish the signature.
   
## generate public key
  the team hold a ceremony to generate public key, each of them provide their secret key and a randum number to generate a polynomial, the secret is const and the degree can decide threshold. 
  The detail of math is the following, player i has secret:````si```` randum num:````ai````, he send ````si + ai*j + ai*j^2```` to player j (if degree is 2), so each play j has ````∑s + ∑a*j + ∑a*j^2````, and if three play do the lagrange they can calculate public key(in elliptic curve form) and secret(or we can call it private key, and they should not do that).
  
## paillier key share
  so, to do this, we need a function that allow two person to generate a key, but both of them doesn't know the key itself, is call multiplication to add. In this situation Allice has a key share A, Bob has B, Allice use paillier encryption (it's likes rsa, but it contain ZK verify) send  ````encryption(A)````, Bob recive the cipher then he mut B to the cipher and also add C to it, then he keep -C as key share and send ````cipher(A*B + C)````to Allice, so after Allice dencrypt the cipher, she got ````A*B + C````, so they can add the key share to get  ````A*B + C +(-C) = A*B````

## make public
  using paillier key share, they can make a function ````f(x) = private key + a1x + a2x**2 ...````, and after doing operating of ecc cruve, they will have ````f(x) = pubkey + a1x*G + (a2x**2)*G ...````, then they can calculate pubkey (although they can calculate private key also but they shouldn't)

## ecdsa
  1  <br>
  2
  
