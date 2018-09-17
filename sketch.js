let xs=[];
let ys=[];

//not sure why this didn't work by putting height-mouseY into ys and mouseX into xs
//for some reason training would make slope and cept turn to NaN, same with mousY into ys and mouseX into xs

//slope and intercept for linear equation(mx+b=y) as mutable tensors
let slope,cept;

//optimizer and learning rate
let lR=.2;
let opt=tf.train.sgd(lR);

//function to guess
function guess(inds){
  //return xs into equation with slope and intercept
  return tf.tensor1d(inds).mul(slope).add(cept);
}

//loss function to optimize(by minimizing)
function loss(gue,act){
  //return meanSquaredError
  return gue.sub(act).square().mean();
}

function setup(){
  createCanvas(innerWidth,innerHeight);
  slope=tf.scalar(random(1)).variable();
  cept=tf.scalar(random(1)).variable();
}

function mousePressed(){
  xs.push(map(mouseX, 0, width, 0, 1));
  ys.push(map(mouseY, 0, height, 1, 0));
}

function draw(){
  tf.tidy(()=>{
    if(xs.length>0){
      //train automatically
      opt.minimize(()=>loss(guess(xs),tf.tensor1d(ys)));
    }
  });
  background(0);
  strokeWeight(16);
  stroke(124,165,57);
  for(let i=0;i<xs.length;i++){
      point(map(xs[i], 0, 1, 0, width),map(ys[i], 0, 1, height, 0));
  }
  //draw prediction line
  strokeWeight(2);
  stroke(165,57,124);
  tf.tidy(()=>{
    //make a guess for y at edges of canvas then get data from tensor1d returned and use to draw line
    let yVals=guess([0,1]).dataSync();
    //change to use dataSync instead of putting line drawing in .then
    //fixes line not showing since data() is async and redrawing background
    line(0,map(yVals[0], 0, 1, height, 0),width,map(yVals[1], 0, 1, height, 0));
  });
}
