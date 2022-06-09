

async function from_js(code) {
  pls = code

  
  await eel.to_python(pls)
  }

from_js()


async function from_js_2(code_2) {
  pls2 = code_2
  
  await eel.to_python_2(pls2)
  }

from_js_2()


async function from_js_3(code_3) {
  pls3 = code_3
  
  await eel.to_python_3(pls3)
  }

from_js_3()



async function button_show_faces(){
  
  var imgg = new Image();
  
  //image.src='../image-2.jpg'
  imgg = '../image-2.jpg'
  var img = document.createElement('img');

  document.querySelector("#image").style.backgroundImage = 'url('+imgg+')';
 
  
  
}

async function button_show_faces_2(){

  //image_2.src='../image-3.jpg'
   var imgg_2 = new Image();
   imgg_2 = '../image-3.jpg'
   

  document.querySelector("#image_2").style.backgroundImage = 'url('+imgg_2+')';
  
  
}



async function button_compare_faces(){

  
  await eel.compare_faces('img/new_2.jpg', 'img/new.jpg')
  
}

async function readTextFile(){
  
  await eel.file_reading()

}

async function video(){
  
  await eel.video_capture()

}



async function analyze() {
    let textarea = document.querySelector('.mytext');
    console.log(textarea.value);
    text = textarea.value

    await eel.analyze_python(text)
}

async function face_rec() {

    await eel.face_recognition_cap()

}
 
async function show_text_js(){
    await eel.show_text_python()
}


eel.expose(js_alarm);
function js_alarm(){

  document.getElementById('Paragraph').innerHTML = "The photo is of the same person!"
  
}


eel.expose(js_alarm_2);
function js_alarm_2(){

  document.getElementById('Paragraph').innerHTML = "Two different people on the photo!"
  
}

var GodObj = {};
var GodObj_2 = {};
var GodObj_3 = {};
var GodObj_4 = {};


eel.expose(age_js);
function age_js(Age, Gender, Race, Emotion){

  a = Age
  b = Gender
  c = Race
  d = Emotion

  GodObj.lala = a;
  GodObj_2.lal = b;
  GodObj_3.la = c;
  GodObj_4.l = d;
  
  return a, b, c, d;

}


async function age(){

  var agee = GodObj.lala;
  var genderr = GodObj_2.lal
  var racee = GodObj_3.la
  var emotionn = GodObj_4.l


  console.log(agee)
  console.log(genderr)
  console.log(racee)

  document.getElementById('Paragraph2').innerHTML = "Age:  " + agee;
  document.getElementById('Paragraph3').innerHTML = "Gender:  " + genderr;
  document.getElementById('Paragraph4').innerHTML = "Race:  " + racee;
  document.getElementById('Paragraph5').innerHTML = "Emotion:  " + emotionn;
 


}
   

  


    

