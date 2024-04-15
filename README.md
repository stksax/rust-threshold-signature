# rust-threshold-signature
   what we need to do for changing the minimum threshold for a team to make a proposal or vote, threshold-signature can do that easily, you just need to change the public key of the group, and it is reusable if the threshold doesn't change. With this project we can detact who has been hack to send the wrong signature for the protocal, and even doesn't need a merkle tree to proof the member is in the group, next I will explame step by step from making a public key, do some secret exchange for the signature, how we verify the message for secret exchange, and detect who has been hack, and finish the signature, the idea comes from ````https://eprint.iacr.org/2020/540.pdf````.
   
## generate public key
  the team hold a ceremony to generate public key, each of them provide their secret key and a randum number to generate a polynomial, the secret is const and the degree can decide threshold. 
  The detail of math is the following, player i has secret:````si```` randum num:````ai````, he send ````si + ai*j + ai*j^2```` to player j (if degree is 2), so each play j has ````∑s + ∑a*j + ∑a*j^2````, and if three play do the lagrange they can calculate public key(in elliptic curve form) and secret(or we can call it private key, and they should not do that), src in ````key_generate.rs````.
  
## ecdsa
  after we have a public key, we can use ecdsa for verify the proposal, but before that they need to cooperate for ecdsa, for prevent someone monopolize other's information after he gets other's message, we use multiplication to add (MTA).
  The detail of math is the following, player has secret ````x```` , chose a randum num ````k````, publish public key ````x*G```` , commitment ````(k⁻¹)*G```` , response ````s = k * ( m + xr )```` (r is x coordinate of elliptic curve point commitment, m is message). Verifier can calculate ````s⁻¹(m*g) + s⁻¹(r*pub_key) === commitment````, src in ````other_small_project/ecdsa.rs````.

## multiplication to add (MTA)
  player1 has A, player2 has B, if they want to mut them without let other knows their secret, player1 send a cipher ````En(A)```` , player2 do mut and add to it ````En(A*B + C)```` (C is a randum num he chose), then player1 decrypt the cipher so he got ````A*B + C```` , player2 hold ````-C```` , so they add this together to get ````A * B```` , src in ````other_small_project/mta.rs````, and how can we make sure other sends the correct secret, I use zkp in ````paillier_verify.rs````.

## multi parties ecdsa 
   we change the ecdsa commitment from ````(k⁻¹)*G```` to ````((k1 + k2 )*(r1 + r2 ))⁻¹ * (r1 + r2) * G```` , response ````s = k * ( m + xr )```` to ````s = (k1 + k2) * ( m + (x1 + x2) r )```` , so here we can calaulate ````k1 * x2```` with MTA and verify, and why do we use ````((k1 + k2 )*(r1 + r2 ))⁻¹ * (r1 + r2) * G```` rather than  ````(k1 + k2)⁻¹ * G````, because they can calculate  ````∑ki * commitment i```` and it should be equal to ````G```` because ````(k1 + k2) * (k1 + k2)⁻¹ * G == G```` , if this step was wrong they can totaly open the randum num ````ki , ri```` to find who has been hack sence the private key havn't been use yet. But if above all correct but in the final step the ecdsa verify fail, we need to do the singal ecdsa ome by one for detect (src in ````main.rs````), that is a little bit Annoying, so I have the other idea using eddsa. 
  
## multi parties eddsa
  since the last step fail we still need to run a singal ecdsa, why we just use eddsa to make this more simple, it just need to run the final step to detect hacker.
  The detail of math is the following, player has secret ````x```` , chose a randum num ````k````, publish public key ````x*G```` , commitment ````k*G```` , response ````s = k + hash_num * x```` (hash_num comes from message). Verifier can calculate ````commitment + hash_num * pub_key == s * g````, src in ````other_small_project/eddsa.rs````, and a team protocal commitment ````(k1 + k2) * G```` , response ````s = (k1 + k2) + hash_num * (x1 + x2)````, and it's easy to verify in singal eddsa, src in ````other_small_project/group_eddsa.rs````.

## get start
  run ````cargo test```` for every testing, run ````cargo run```` for the main protocal.
