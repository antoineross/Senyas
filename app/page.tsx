import { BlogPosts } from 'app/components/posts'

export default function Page() {
  return (
    <section>
      <h1 className="mb-8 text-2xl font-semibold tracking-tighter">
        Senyas
      </h1>
      <p className="mb-4">
        we implemented a <strong>filipino sign language recognition model</strong> for both static and dynamic hand gestures using <strong>mediapipe</strong>.
      </p>
      <p className="mb-4">
      the goal is simple: we wanted to turn all sign language gestures into text through a camera feed.
        in simple terms, the pipeline goes like this: sign language gesture → model prediction → text.
      </p>
      <p className="mb-4">
        if you're keen on trying it out for yourself, you can check it out <a className="text-blue-500 underline" href="https://senyas.vercel.app/fsl" target="_blank">here</a>.
        we also write some articles on the process of building it, you can check it out <a className="text-blue-500 underline" href="https://senyas.vercel.app/blog" target="_blank">here</a>.
        </p>
      {/* <p className="mb-4">
        if we look under the hood, it goes more like this: videos are flattened into a 1d vector (3d → 1d) → vector passed into mediapipe → mediapipe predictions are passed into the model → model makes prediction → prediction reflects as text on the device.
      </p> */}
      <div className="my-8">
        <BlogPosts />
      </div>
      </section>
  )
}
